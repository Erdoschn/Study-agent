from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


BLOCKED_NAMES = {".env", ".env.local", ".env.production", ".git-credentials", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea"}


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        p = Path(directory) / name
        name_key = name.casefold()
        if name_key in SKIP_DIRS or name_key in BLOCKED_NAMES or p.suffix.casefold() in BLOCKED_SUFFIXES:
            ignored.add(name)
    return ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("python", "pytest"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    source = Path("/workspace")
    work = Path("/sandbox")
    work.mkdir(parents=True, exist_ok=True)
    for child in work.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(source, work, ignore=_ignore, symlinks=True, dirs_exist_ok=True)

    os.chdir(work)
    sys.path.insert(0, str(work))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HOME"] = "/tmp"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    for raw_path in args.paths:
        path = raw_path.replace("\\", "/")
        parts = [p for p in path.split("/") if p not in {"", "."}]
        if (
            not parts
            or any(p == ".." for p in parts)
            or any(p.startswith("-") for p in parts)
            or ":" in raw_path
            or "::" in raw_path
            or path.startswith("/")
        ):
            return 125

    if args.kind == "python":
        if len(args.paths) != 1:
            return 125
        target = (work / args.paths[0]).resolve()
        if not target.is_file() or not str(target).startswith(str(work) + os.sep):
            return 125
        import runpy
        runpy.run_path(str(target), run_name="__main__")
        return 0

    import pytest
    for raw_path in args.paths:
        target = (work / raw_path).resolve()
        try:
            target.relative_to(work)
        except ValueError:
            return 125
        if not target.is_file() or target.suffix.casefold() != ".py":
            return 125
    argv = ["-q", "--disable-warnings", "--maxfail=5"]
    argv.extend(args.paths)
    return int(pytest.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
