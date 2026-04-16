"""RBAC tests — role hierarchy, permissions, admin/reviewer resolution."""

from __future__ import annotations


def test_role_hierarchy_admin_implies_reviewer(monkeypatch):
    from engine.auth import User

    admin = User(username="alice", role="admin", authed_at=0.0)
    assert admin.is_admin
    assert admin.is_reviewer           # admin inherits reviewer permissions
    assert admin.role_at_least("reviewer")
    assert admin.role_at_least("analyst")


def test_role_hierarchy_analyst_cannot_review(monkeypatch):
    from engine.auth import User
    analyst = User(username="bob", role="analyst", authed_at=0.0)
    assert not analyst.is_reviewer
    assert not analyst.is_admin
    assert not analyst.role_at_least("reviewer")
    assert analyst.role_at_least("analyst")


def test_permission_map_gates_file_disposition(monkeypatch):
    from engine.auth import User

    analyst  = User(username="a", role="analyst",  authed_at=0.0)
    reviewer = User(username="r", role="reviewer", authed_at=0.0)
    admin    = User(username="x", role="admin",    authed_at=0.0)

    assert analyst.can("view_alerts")
    assert analyst.can("download_sar")
    assert not analyst.can("file_disposition")     # analysts can't dispo
    assert reviewer.can("file_disposition")
    assert not reviewer.can("edit_watchlist")      # only admin
    assert admin.can("edit_watchlist")


def test_permission_unknown_action_defaults_to_allow():
    from engine.auth import User
    u = User(username="u", role="analyst", authed_at=0.0)
    assert u.can("some_new_action_not_in_map") is True


def test_admin_username_resolution(monkeypatch):
    monkeypatch.setenv("AML_ADMIN_USERNAMES",    "root,ops")
    monkeypatch.setenv("AML_REVIEWER_USERNAMES", "ops,qa")  # ops is in both
    from engine.auth import _resolve_role
    # Admin list takes precedence
    assert _resolve_role("root")    == "admin"
    assert _resolve_role("ops")     == "admin"
    assert _resolve_role("qa")      == "reviewer"
    assert _resolve_role("nobody")  == "analyst"
