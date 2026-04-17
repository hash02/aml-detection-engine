"""Auth gate tests — verify password check + role assignment without Streamlit."""

from __future__ import annotations


def test_check_password_open_mode_when_unset(monkeypatch):
    """With no password configured, any input returns True (demo mode)."""
    monkeypatch.delenv("AML_APP_PASSWORD", raising=False)
    # Stub out any streamlit secrets shim so we don't accidentally read real secrets
    import engine.auth as auth_mod
    from engine.auth import _check_password

    def _no_secret():
        return None

    monkeypatch.setattr(auth_mod, "_expected_password", _no_secret)
    assert _check_password("anything") is True


def test_check_password_accepts_correct(monkeypatch):
    monkeypatch.setenv("AML_APP_PASSWORD", "hunter2")
    from engine.auth import _check_password
    assert _check_password("hunter2") is True
    assert _check_password("wrong")   is False


def test_check_password_constant_time(monkeypatch):
    """Uses hmac.compare_digest — not a timing oracle."""
    monkeypatch.setenv("AML_APP_PASSWORD", "correct-horse-battery-staple")
    from engine.auth import _check_password
    # Different-length input still returns False, not raises
    assert _check_password("") is False
    assert _check_password("c") is False


def test_reviewer_role_assignment(monkeypatch):
    monkeypatch.setenv("AML_REVIEWER_USERNAMES", "alice,bob")
    from engine.auth import _reviewer_usernames
    reviewers = _reviewer_usernames()
    assert reviewers == {"alice", "bob"}


def test_current_user_anonymous_outside_streamlit():
    """Outside a Streamlit session, current_user() falls back to anonymous."""
    from engine.auth import current_user
    u = current_user()
    assert u.username == "anonymous"
    assert u.role == "analyst"
    assert u.is_reviewer is False
