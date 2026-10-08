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
        if name in SKIP_DIRS or name in BLOCKED_NAMES or p.suffix.casefold() in BLOCKED_SUFFIXES:
            ignored.add(name)
    return ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("python", "pytest"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    source = Path("/workspace")
    work = Path("/sandbox")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(source, work, ignore=_ignore)

    os.chdir(work)
    sys.path.insert(0, str(work))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HOME"] = "/tmp/home"

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
    argv = ["-q", "--disable-warnings", "--maxfail=5"]
    argv.extend(args.paths)
    return int(pytest.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
