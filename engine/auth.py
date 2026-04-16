"""
engine/auth.py — Minimal Streamlit auth gate
=============================================

Password-based gate backed by `AML_APP_PASSWORD` (env) or
`st.secrets["AML_APP_PASSWORD"]` (Streamlit Cloud). Designed to keep the
PR scope tight: one password, two roles (analyst / reviewer), session-
scoped (no persistent user database).

When `AML_APP_PASSWORD` is unset, the gate is bypassed and every user is
treated as an anonymous analyst — identical to the current public demo.

Usage in streamlit_app.py:
    from engine.auth import require_auth, current_user
    require_auth()   # short-circuits with st.stop() if not yet logged in
    user = current_user()
    if user.role == 'reviewer':
        show_review_controls()
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass

# v13 RBAC — roles are a strict hierarchy: admin > reviewer > analyst.
# Any code path that needs "at least reviewer" should check
# `user.role_at_least("reviewer")`, never `user.role == "reviewer"`,
# so admins are implicitly allowed everywhere a reviewer is.
ROLE_ORDER = {"analyst": 0, "reviewer": 1, "admin": 2}

# Per-role allow lists. Routes not listed are always permitted.
PERMISSIONS = {
    "view_alerts":       "analyst",
    "download_sar":      "analyst",
    "file_disposition":  "reviewer",   # escalate / dismiss / sar_filed
    "edit_watchlist":    "admin",
    "view_audit_log":    "reviewer",
    "refresh_feeds":     "admin",
}


@dataclass(frozen=True)
class User:
    username: str
    role: str           # "analyst" | "reviewer" | "admin"
    authed_at: float

    @property
    def is_reviewer(self) -> bool:
        return self.role_at_least("reviewer")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def role_at_least(self, required: str) -> bool:
        """True if this user's role is at least `required` in the hierarchy."""
        return ROLE_ORDER.get(self.role, -1) >= ROLE_ORDER.get(required, 999)

    def can(self, action: str) -> bool:
        """Check a named permission. Unknown actions default to allow."""
        needed = PERMISSIONS.get(action)
        if needed is None:
            return True
        return self.role_at_least(needed)


def _expected_password() -> str | None:
    """Pull password from env first, then st.secrets if available."""
    pw = os.environ.get("AML_APP_PASSWORD")
    if pw:
        return pw
    try:
        import streamlit as st
        return st.secrets.get("AML_APP_PASSWORD")  # type: ignore[attr-defined]
    except Exception:
        return None


def _reviewer_usernames() -> set[str]:
    raw = os.environ.get("AML_REVIEWER_USERNAMES", "")
    if not raw:
        try:
            import streamlit as st
            raw = st.secrets.get("AML_REVIEWER_USERNAMES", "") or ""  # type: ignore[attr-defined]
        except Exception:
            raw = ""
    return {u.strip() for u in raw.split(",") if u.strip()}


def _admin_usernames() -> set[str]:
    raw = os.environ.get("AML_ADMIN_USERNAMES", "")
    if not raw:
        try:
            import streamlit as st
            raw = st.secrets.get("AML_ADMIN_USERNAMES", "") or ""  # type: ignore[attr-defined]
        except Exception:
            raw = ""
    return {u.strip() for u in raw.split(",") if u.strip()}


def _resolve_role(username: str) -> str:
    """Admins override reviewers; otherwise fall back to analyst."""
    if username in _admin_usernames():
        return "admin"
    if username in _reviewer_usernames():
        return "reviewer"
    return "analyst"


def _check_password(submitted: str) -> bool:
    expected = _expected_password()
    if not expected:
        return True  # no password configured → open demo mode
    return hmac.compare_digest(
        hashlib.sha256(submitted.encode()).hexdigest(),
        hashlib.sha256(expected.encode()).hexdigest(),
    )


def require_auth() -> User:
    """Block the Streamlit page until the user has authenticated.

    Returns the authenticated User. If no password is configured,
    returns an anonymous analyst user immediately.
    """
    import streamlit as st

    expected = _expected_password()
    if not expected:
        # Open-demo path — no gate
        return User(username="anonymous", role="analyst", authed_at=time.time())

    if st.session_state.get("_auth_user"):
        return st.session_state["_auth_user"]

    with st.sidebar:
        st.markdown("#### 🔐 Sign in")
        username = st.text_input("Username", key="_auth_username")
        password = st.text_input("Password", type="password", key="_auth_password")
        if st.button("Sign in", use_container_width=True):
            if not username:
                st.error("Username required")
            elif not _check_password(password):
                st.error("Invalid password")
            else:
                user = User(
                    username=username,
                    role=_resolve_role(username),
                    authed_at=time.time(),
                )
                st.session_state["_auth_user"] = user
                st.rerun()

    st.info("Please sign in via the sidebar to use the detection engine.")
    st.stop()


def current_user() -> User:
    """Return the current session user, or an anonymous analyst."""
    try:
        import streamlit as st
        u = st.session_state.get("_auth_user")
        if u:
            return u
    except Exception:
        pass
    return User(username="anonymous", role="analyst", authed_at=time.time())
