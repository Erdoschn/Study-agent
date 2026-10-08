from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Iterable


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

    def run(self, kind: str, paths: Iterable[str] = ()) -> SandboxResult:
        if kind not in {"python", "pytest"}:
            raise ValueError("只支持 python / pytest。")
        safe_paths = [str(p) for p in paths]
        if kind == "python" and len(safe_paths) != 1:
            raise ValueError("RUN_PYTHON 需要且只能需要一个 script_path。")

        name = "coder-sandbox-" + uuid.uuid4().hex[:20]
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
            f"type=bind,src={self.workspace},dst=/workspace,readonly",
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
            raise RuntimeError(
                "安全执行被拒绝：未找到 docker CLI。"
            ) from exc

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
