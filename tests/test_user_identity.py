from __future__ import annotations

import pytest

from core.user_identity import UserProfileLimitError, normalize_username, user_graph_path


@pytest.mark.parametrize(
    ("value", "expected"),
    [("  Alice_01 ", "alice_01"), ("Ａlice", "alice"), ("张三", "张三")],
)
def test_username_normalization(value, expected):
    assert normalize_username(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", " ", "../alice", "a/b", r"a\\b", "bad name", ".", "..", "a:b",
     "x\n", "x\r", "x\t", "alice\x00", "x\x7f"],
)
def test_username_rejects_empty_or_unsafe_values(value):
    with pytest.raises(ValueError):
        normalize_username(value)


def test_username_rejects_non_string_and_oversized_values():
    with pytest.raises(ValueError):
        normalize_username(None)
    with pytest.raises(ValueError):
        normalize_username("a" * 65)


def test_each_user_gets_an_isolated_storage_path(tmp_path):
    alice = user_graph_path("Alice", data_root=tmp_path, max_profiles=3)
    bob = user_graph_path("Bob", data_root=tmp_path, max_profiles=3)
    assert alice != bob
    assert alice == user_graph_path("alice", data_root=tmp_path, max_profiles=3)
    assert alice.name == "knowledge_graph.sqlite3"
    assert "alice" not in str(alice).lower()
    assert alice.parent.parent.name == "users"


def test_profile_limit_is_enforced(tmp_path):
    user_graph_path("alice", data_root=tmp_path, max_profiles=1)
    with pytest.raises(UserProfileLimitError):
        user_graph_path("bob", data_root=tmp_path, max_profiles=1)
