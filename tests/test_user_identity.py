from __future__ import annotations

import pytest

from core.user_identity import (
    UserProfileLimitError,
    normalize_username,
    user_graph_path,
)


def test_username_normalization_is_case_insensitive_and_unicode_normalized():
    assert normalize_username("  Alice_01 ") == "alice_01"
    assert normalize_username("Ａlice") == "alice"
    assert normalize_username("张三") == "张三"


@pytest.mark.parametrize("value", ["", " ", "../alice", "a/b", r"a\\b", "bad name", ".", "..", "a:b", "x\n"])
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
    alice_again = user_graph_path("alice", data_root=tmp_path, max_profiles=3)

    assert alice != bob
    assert alice == alice_again
    assert alice.name == "knowledge_graph.sqlite3"
    assert "alice" not in str(alice).lower()
    assert alice.parent.parent.name == "users"


def test_profile_limit_is_enforced(tmp_path):
    user_graph_path("alice", data_root=tmp_path, max_profiles=1)
    with pytest.raises(UserProfileLimitError):
        user_graph_path("bob", data_root=tmp_path, max_profiles=1)
