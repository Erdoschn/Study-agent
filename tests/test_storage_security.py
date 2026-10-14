from __future__ import annotations

import pytest

from coder.knowledge_graph import CoderKnowledgeGraph
from core.knowledge_graph import KnowledgeGraph
from core.storage_security import ensure_private_directory, ensure_safe_sqlite_path


def _symlink_or_skip(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable in this environment")


def test_sqlite_guard_creates_a_private_regular_file(tmp_path):
    path = ensure_safe_sqlite_path(tmp_path / "graph.sqlite3")

    assert path.is_file()
    assert not path.is_symlink()
    if __import__("os").name != "nt":
        assert path.stat().st_mode & 0o077 == 0


def test_sqlite_guard_rejects_database_symlinks_without_touching_target(tmp_path):
    target = tmp_path / "external.sqlite3"
    target.write_bytes(b"do-not-overwrite")
    link = tmp_path / "graph.sqlite3"
    _symlink_or_skip(link, target)

    with pytest.raises(PermissionError, match="符号链接"):
        ensure_safe_sqlite_path(link)

    assert target.read_bytes() == b"do-not-overwrite"


def test_sqlite_guard_rejects_symlinked_sidecars(tmp_path):
    path = ensure_safe_sqlite_path(tmp_path / "graph.sqlite3")
    target = tmp_path / "external-wal"
    target.write_bytes(b"private")
    _symlink_or_skip(tmp_path / "graph.sqlite3-wal", target)

    with pytest.raises(PermissionError, match="符号链接"):
        ensure_safe_sqlite_path(path)

    assert target.read_bytes() == b"private"


def test_study_agent_knowledge_graph_rejects_symlinked_database(tmp_path):
    target = tmp_path / "external.sqlite3"
    target.write_bytes(b"not a database")
    path = tmp_path / "knowledge.sqlite3"
    _symlink_or_skip(path, target)

    with pytest.raises(PermissionError, match="符号链接"):
        KnowledgeGraph(str(path))

    assert target.read_bytes() == b"not a database"


def test_coder_knowledge_graph_rejects_symlinked_database(tmp_path):
    target = tmp_path / "external.sqlite3"
    target.write_bytes(b"not a database")
    _symlink_or_skip(tmp_path / ".coder-knowledge.sqlite3", target)

    with pytest.raises(PermissionError, match="符号链接"):
        CoderKnowledgeGraph(tmp_path)

    assert target.read_bytes() == b"not a database"


def test_private_directory_uses_restricted_permissions(tmp_path):
    path = ensure_private_directory(tmp_path / "browser-profile")

    assert path.is_dir()
    assert not path.is_symlink()
    if __import__("os").name != "nt":
        assert path.stat().st_mode & 0o077 == 0


def test_private_directory_rejects_symlinks_and_non_directories(tmp_path):
    target = tmp_path / "real-profile"
    target.mkdir()
    link = tmp_path / "profile-link"
    _symlink_or_skip(link, target)

    with pytest.raises(PermissionError, match="符号链接"):
        ensure_private_directory(link)

    with pytest.raises(PermissionError, match="专用子目录"):
        ensure_private_directory(".")


def test_sqlite_guard_rejects_symlinked_parent_directories(tmp_path):
    target = tmp_path / "external-parent"
    target.mkdir()
    link = tmp_path / "linked-parent"
    _symlink_or_skip(link, target)

    with pytest.raises(PermissionError, match="符号链接"):
        ensure_safe_sqlite_path(link / "graph.sqlite3")

    with pytest.raises(PermissionError, match="符号链接"):
        ensure_private_directory(link / "browser-profile")

    assert list(target.iterdir()) == []
