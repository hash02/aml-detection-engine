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


@dataclass(frozen=True)
class User:
    username: str
    role: str           # "analyst" | "reviewer"
    authed_at: float

    @property
    def is_reviewer(self) -> bool:
        return self.role == "reviewer"


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
                role = "reviewer" if username in _reviewer_usernames() else "analyst"
                user = User(username=username, role=role, authed_at=time.time())
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
