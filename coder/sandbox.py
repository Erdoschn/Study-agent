from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Iterable


READABLE_EXTENSIONS = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".json", ".toml",
    ".ini", ".cfg", ".yaml", ".yml", ".csv", ".tsv", ".xml",
}
BLOCKED_NAMES = {
    ".env", ".env.local", ".env.production", ".git-credentials",
    "id_rsa", "id_ed25519",
}
BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea"}


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_limited: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.output_limited


class DockerPythonSandbox:
    IMAGE = "study-agent-coder-python:1"
    MAX_OUTPUT_BYTES = 65_536

    def __init__(
        self,
        workspace: str | Path,
        *,
        timeout_seconds: float = 120,
        memory: str = "768m",
        cpus: str = "2.0",
        pids: int = 128,
    ):
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.memory = memory
        self.cpus = cpus
        self.pids = max(16, int(pids))

    def preflight(self) -> None:
        try:
            probe = subprocess.run(
                ["docker", "image", "inspect", self.IMAGE],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("安全执行被拒绝：未找到 docker CLI。") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("安全执行被拒绝：docker preflight 超时。") from exc
        if probe.returncode != 0:
            detail = probe.stderr.decode("utf-8", errors="replace").strip()[:300]
            raise RuntimeError(
                f"安全执行被拒绝：未找到本地 Coder 沙箱镜像 {self.IMAGE}。"
                + (f" {detail}" if detail else "")
            )

    @staticmethod
    def _is_special(path: Path) -> bool:
        try:
            st = path.lstat()
        except FileNotFoundError:
            return False
        return bool(
            stat.S_ISLNK(st.st_mode)
            or (os.name == "nt" and getattr(st, "st_file_attributes", 0) & 0x400)
        )

    def _stage_workspace(self) -> tempfile.TemporaryDirectory:
        stage = tempfile.TemporaryDirectory(
            prefix=".coder-sandbox-",
            dir=str(self.workspace),
        )
        root = Path(stage.name)
        for current, dirs, files in os.walk(self.workspace, topdown=True, followlinks=False):
            current_path = Path(current)
            dirs[:] = [
                name for name in dirs
                if name not in SKIP_DIRS
                and name not in BLOCKED_NAMES
                and not name.startswith(".coder-sandbox-")
                and not self._is_special(current_path / name)
            ]
            relative_dir = current_path.relative_to(self.workspace)
            target_dir = root / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in files:
                source = current_path / name
                if self._is_special(source):
                    continue
                if name in BLOCKED_NAMES or name.startswith(".coder-sandbox-"):
                    continue
                if source.suffix.casefold() in BLOCKED_SUFFIXES:
                    continue
                if source.suffix.casefold() not in READABLE_EXTENSIONS:
                    continue
                destination = target_dir / name
                if source.stat().st_size > 1_048_576:
                    continue
                shutil.copyfile(source, destination)
        return stage

    def run(self, kind: str, paths: Iterable[str] = ()) -> SandboxResult:
        if kind not in {"python", "pytest"}:
            raise ValueError("只支持 python / pytest。")
        safe_paths = [str(p) for p in paths]
        for path in safe_paths:
            normalized = path.replace("\\", "/")
            parts = [p for p in normalized.split("/") if p not in {"", "."}]
            if (
                not parts
                or any(p == ".." for p in parts)
                or any(p.startswith("-") for p in parts)
                or path.startswith(("/", "\\"))
                or ":" in path
                or "::" in path
            ):
                raise PermissionError("sandbox 路径非法。")
        if kind == "python" and len(safe_paths) != 1:
            raise ValueError("RUN_PYTHON 需要且只能需要一个 script_path。")

        name = "coder-sandbox-" + uuid.uuid4().hex[:20]
        with self._stage_workspace() as staged:
            command = [
                "docker", "run", "--rm", "--init", "--pull=never",
                "--name", name,
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
                "--pids-limit", str(self.pids),
                "--memory", self.memory,
                "--cpus", self.cpus,
                "--ulimit", "nofile=256:256",
                "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=128m",
                "--tmpfs", "/sandbox:rw,nosuid,nodev,noexec,size=512m",
                "--mount",
                f"type=bind,src={staged},dst=/workspace,readonly",
                self.IMAGE,
                kind,
                *safe_paths,
            ]
            return self._run_limited(command, name)

    def _run_limited(self, command: list[str], name: str) -> SandboxResult:
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("安全执行被拒绝：未找到 docker CLI。") from exc

        limit = self.MAX_OUTPUT_BYTES
        stdout = bytearray()
        stderr = bytearray()
        exceeded = Event()

        def reader(stream, buffer):
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                if len(buffer) < limit:
                    buffer.extend(chunk[: limit - len(buffer)])
                if len(buffer) >= limit:
                    exceeded.set()

        threads = [
            Thread(target=reader, args=(proc.stdout, stdout), daemon=True),
            Thread(target=reader, args=(proc.stderr, stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()

        started = time.monotonic()
        timed_out = False
        while proc.poll() is None:
            if exceeded.is_set():
                proc.kill()
                break
            if time.monotonic() - started >= self.timeout_seconds:
                timed_out = True
                proc.kill()
                break
            time.sleep(0.05)

        returncode = proc.wait()
        for thread in threads:
            thread.join(timeout=1.0)
        if timed_out or exceeded.is_set():
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=10,
            )
        return SandboxResult(
            returncode=returncode,
            stdout=bytes(stdout).decode("utf-8", errors="replace"),
            stderr=bytes(stderr).decode("utf-8", errors="replace"),
            timed_out=timed_out,
            output_limited=exceeded.is_set(),
        )
