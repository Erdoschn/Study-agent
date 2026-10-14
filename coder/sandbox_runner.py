from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


BLOCKED_NAMES = {".env", ".env.local", ".env.production", ".git-credentials", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".coder-backup"}
READABLE_EXTENSIONS = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".json", ".toml",
    ".ini", ".cfg", ".yaml", ".yml", ".csv", ".tsv", ".xml", ".ipynb",
    ".html", ".htm", ".css", ".js", ".mjs", ".cjs", ".c", ".h", ".cc",
    ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".java", ".properties",
    ".gradle", ".kts", ".sql", ".svg",
}
SPECIAL_TEXT_FILENAMES = {"makefile", "gnumakefile", "dockerfile", "containerfile"}
MAX_PROCESS_OUTPUT_BYTES = 48 * 1024
MAX_ARGS = 16
MAX_ARGUMENT_BYTES = 4096
MAX_STDIN_BYTES = 16 * 1024
PROJECT_SUFFIXES = {
    "html": {".html", ".htm"},
    "cpp": {".cc", ".cpp", ".cxx"},
    "java": {".java"},
}
JS_TYPES = {
    "", "module", "text/javascript", "application/javascript",
    "text/ecmascript", "application/ecmascript",
}


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        item = Path(directory) / name
        name_key = name.casefold()
        if (
            name_key in SKIP_DIRS
            or name_key in BLOCKED_NAMES
            or item.suffix.casefold() in BLOCKED_SUFFIXES
        ):
            ignored.add(name)
        elif item.is_dir() and not item.is_symlink():
            # Preserve ordinary source directories; only filter files by suffix.
            continue
        elif item.suffix.casefold() not in READABLE_EXTENSIONS and name_key not in SPECIAL_TEXT_FILENAMES:
            ignored.add(name)
    return ignored


def _safe_target(work: Path, raw_path: str, suffixes: set[str] | None = None) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("目标路径不能为空。")
    normalized = raw_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if (
        not parts
        or any(part == ".." or part.startswith("-") for part in parts)
        or normalized.startswith("/")
        or ":" in normalized
        or "::" in normalized
    ):
        raise ValueError("目标路径非法。")
    target = (work / normalized).resolve()
    try:
        target.relative_to(work.resolve())
    except ValueError as exc:
        raise ValueError("目标路径越出沙箱工作区。") from exc
    if not target.is_file():
        raise FileNotFoundError(f"找不到源文件：{normalized}")
    if suffixes is not None and target.suffix.casefold() not in suffixes:
        raise ValueError(f"源文件扩展名无效：{normalized}")
    return target


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run_bounded(
    command: list[str],
    *,
    cwd: Path,
    input_text: str = "",
    timeout_seconds: float = 45.0,
) -> tuple[int, str, str, bool, bool]:
    """Run fixed argv without a shell; limit execution time and combined output."""
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE if input_text else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        return 127, "", f"缺少执行工具：{command[0]}（{exc}）", False, False

    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    output_lock = threading.Lock()
    output_size = [0]
    exceeded = threading.Event()

    def read_stream(stream, key):
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                with output_lock:
                    remaining = MAX_PROCESS_OUTPUT_BYTES - output_size[0]
                    if remaining > 0:
                        kept = chunk[:remaining]
                        outputs[key].extend(kept)
                        output_size[0] += len(kept)
                    if len(chunk) > max(remaining, 0):
                        exceeded.set()
        except (OSError, ValueError):
            return

    readers = [
        threading.Thread(target=read_stream, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=read_stream, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for thread in readers:
        thread.start()

    writer = None
    if input_text and proc.stdin is not None:
        payload = input_text.encode("utf-8")

        def write_input():
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()

    started = time.monotonic()
    timed_out = False
    while proc.poll() is None:
        if exceeded.is_set():
            _kill_process_tree(proc)
            break
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            _kill_process_tree(proc)
            break
        time.sleep(0.025)
    returncode = proc.wait()
    for thread in readers:
        thread.join(timeout=1.0)
    if writer is not None:
        writer.join(timeout=0.2)
    return (
        returncode,
        outputs["stdout"].decode("utf-8", errors="replace"),
        outputs["stderr"].decode("utf-8", errors="replace"),
        timed_out,
        exceeded.is_set(),
    )


class _HTMLInspector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.resources: list[tuple[str, bool, bool]] = []
        self.inline_scripts: list[tuple[str, str]] = []
        self._script_type: str | None = None
        self._script_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = {str(key).casefold(): value for key, value in attrs}
        element_id = values.get("id")
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)

        tag = tag.casefold()
        script_type = str(values.get("type") or "").strip().casefold()
        src = values.get("src")
        if src:
            self.resources.append((str(src), tag == "script" and script_type in JS_TYPES, script_type == "module"))

        if tag == "link":
            rel = {item.casefold() for item in str(values.get("rel") or "").split()}
            href = values.get("href")
            if href and rel.intersection({"stylesheet", "icon", "preload", "modulepreload"}):
                check_js = str(values.get("as") or "").casefold() == "script"
                self.resources.append((str(href), check_js, str(values.get("type") or "").casefold() == "module"))

        if tag == "script" and not src and script_type in JS_TYPES:
            self._script_type = script_type
            self._script_parts = []

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.casefold() == "script":
            self._finish_script()

    def handle_data(self, data):
        if self._script_type is not None:
            self._script_parts.append(data)

    def handle_endtag(self, tag):
        if tag.casefold() == "script":
            self._finish_script()

    def _finish_script(self):
        if self._script_type is not None:
            self.inline_scripts.append((self._script_type, "".join(self._script_parts)))
        self._script_type = None
        self._script_parts = []


def _resolve_local_resource(work: Path, html_path: Path, raw_url: str) -> Path | None:
    value = str(raw_url or "").strip()
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    relative = unquote(parsed.path)
    candidate = work / relative.lstrip("/") if relative.startswith("/") else html_path.parent / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(work.resolve())
    except ValueError:
        return None
    return resolved


def _node_check(path: Path, *, is_module: bool, work: Path):
    check_path = path
    temporary_path = None
    if is_module and path.suffix.casefold() != ".mjs":
        temporary_path = Path("/build") / ("module-check-" + str(abs(hash(str(path)))) + ".mjs")
        shutil.copyfile(path, temporary_path)
        check_path = temporary_path
    code, stdout, stderr, timed_out, output_limited = _run_bounded(
        ["node", "--check", str(check_path)],
        cwd=work,
        timeout_seconds=20.0,
    )
    if temporary_path is not None:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return code, stdout, stderr, timed_out, output_limited


def _run_html(work: Path, target: Path):
    inspector = _HTMLInspector()
    try:
        inspector.feed(target.read_text(encoding="utf-8-sig"))
        inspector.close()
    except (OSError, UnicodeError, ValueError) as exc:
        return 1, "", f"HTML 解析失败：{type(exc).__name__}: {exc}", False, False

    errors = [f"重复 id：{value}" for value in sorted(inspector.duplicate_ids)]
    js_files: list[tuple[Path, bool, str]] = []
    checked = set()
    for raw_url, check_js, is_module in inspector.resources:
        resource = _resolve_local_resource(work, target, raw_url)
        if resource is None:
            continue
        if not resource.is_file():
            errors.append(f"找不到本地资源：{raw_url}")
            continue
        if check_js and resource.suffix.casefold() in {".js", ".mjs", ".cjs"}:
            marker = (str(resource), is_module)
            if marker not in checked:
                checked.add(marker)
                js_files.append((resource, is_module, raw_url))
    if errors:
        return 1, "", "HTML 检查失败：\n- " + "\n- ".join(errors), False, False

    successful_scripts = 0
    for script_path, is_module, label in js_files:
        code, stdout, stderr, timed_out, output_limited = _node_check(
            script_path, is_module=is_module, work=work
        )
        if timed_out or output_limited:
            return code, stdout, stderr, timed_out, output_limited
        if code != 0:
            return 1, stdout, f"JavaScript 语法检查失败（{label}）：\n{stderr}", False, False
        successful_scripts += 1

    for index, (script_type, script_text) in enumerate(inspector.inline_scripts, 1):
        if not script_text.strip():
            continue
        script_path = Path("/build") / f"inline-script-{index}{'.mjs' if script_type == 'module' else '.js'}"
        script_path.write_text(script_text, encoding="utf-8")
        code, stdout, stderr, timed_out, output_limited = _node_check(
            script_path, is_module=script_type == "module", work=work
        )
        if timed_out or output_limited:
            return code, stdout, stderr, timed_out, output_limited
        if code != 0:
            return 1, stdout, f"JavaScript 语法检查失败（inline script #{index}）：\n{stderr}", False, False
        successful_scripts += 1

    return (
        0,
        f"HTML 检查通过：已解析入口页，检查了 {len(inspector.resources)} 个本地资源引用、"
        f"{len(inspector.ids)} 个唯一 id，以及 {successful_scripts} 段 JavaScript 语法。\n",
        "",
        False,
        False,
    )


def _validate_project_request(work: Path, request: object):
    if not isinstance(request, dict) or set(request) != {"kind", "path", "sources", "args", "stdin"}:
        raise ValueError("RUN_PROJECT 请求结构无效。")
    kind = str(request["kind"] or "").casefold()
    if kind not in PROJECT_SUFFIXES:
        raise ValueError("RUN_PROJECT 只支持 html / cpp / java。")
    target = _safe_target(work, request["path"], PROJECT_SUFFIXES[kind])
    sources = request["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 50:
        raise ValueError("RUN_PROJECT.sources 必须是 1 到 50 个相对路径。")
    resolved_sources = []
    seen = set()
    for raw in sources:
        source = _safe_target(work, raw, PROJECT_SUFFIXES[kind])
        marker = str(source)
        if marker not in seen:
            resolved_sources.append(source)
            seen.add(marker)
    if str(target) not in seen:
        resolved_sources.insert(0, target)
    if kind == "html" and len(resolved_sources) != 1:
        raise ValueError("HTML 检查只接受一个入口页面。")
    args = request["args"]
    if (
        not isinstance(args, list)
        or len(args) > MAX_ARGS
        or any(not isinstance(item, str) or "\0" in item for item in args)
        or sum(len(item.encode("utf-8")) for item in args) > MAX_ARGUMENT_BYTES
    ):
        raise ValueError("RUN_PROJECT.args 无效或超过安全上限。")
    stdin_text = request["stdin"]
    if not isinstance(stdin_text, str) or len(stdin_text.encode("utf-8")) > MAX_STDIN_BYTES:
        raise ValueError("RUN_PROJECT.stdin 必须是最多 16 KiB 的文本。")
    if kind == "html" and (args or stdin_text):
        raise ValueError("HTML 检查不接受程序参数或标准输入。")
    return kind, target, resolved_sources, args, stdin_text


def _run_project(work: Path, request: object):
    try:
        kind, target, sources, args, stdin_text = _validate_project_request(work, request)
    except Exception as exc:
        return 125, "", f"RUN_PROJECT 请求无效：{type(exc).__name__}: {exc}", False, False

    if kind == "html":
        return _run_html(work, target)

    if kind == "cpp":
        command = [
            "g++", "-std=c++17", "-O2", "-pipe", "-Wall", "-Wextra",
            "-o", "/build/coder-program",
            *[str(path.relative_to(work)) for path in sources],
        ]
        code, compile_out, compile_err, timed_out, limited = _run_bounded(
            command, cwd=work, timeout_seconds=60.0
        )
        if timed_out or limited or code != 0:
            return code, compile_out, compile_err, timed_out, limited
        run_code, run_out, run_err, run_timed_out, run_limited = _run_bounded(
            ["/build/coder-program", *args], cwd=work,
            input_text=stdin_text, timeout_seconds=45.0
        )
        return run_code, run_out, "\n".join(x for x in (compile_err, run_err) if x), run_timed_out, run_limited

    package_match = re.search(
        r"(?m)^\s*package\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*;",
        target.read_text(encoding="utf-8-sig"),
    )
    main_class = target.stem
    if package_match:
        main_class = package_match.group(1) + "." + main_class
    Path("/build/classes").mkdir(parents=True, exist_ok=True)
    compile_command = [
        "javac", "-J-Xmx384m", "-J-XX:+UseSerialGC", "-encoding", "UTF-8",
        "-d", "/build/classes", *[str(path.relative_to(work)) for path in sources],
    ]
    code, compile_out, compile_err, timed_out, limited = _run_bounded(
        compile_command, cwd=work, timeout_seconds=60.0
    )
    if timed_out or limited or code != 0:
        return code, compile_out, compile_err, timed_out, limited
    run_code, run_out, run_err, run_timed_out, run_limited = _run_bounded(
        ["java", "-Xmx384m", "-XX:+UseSerialGC", "-cp", "/build/classes", main_class, *args],
        cwd=work, input_text=stdin_text, timeout_seconds=45.0
    )
    return run_code, run_out, "\n".join(x for x in (compile_err, run_err) if x), run_timed_out, run_limited


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("python", "pytest", "project"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    project_request = None
    if args.kind == "project":
        try:
            project_request = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"RUN_PROJECT 请求 JSON 无效：{exc}", file=sys.stderr)
            return 125

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

    if args.kind == "project":
        code, stdout, stderr, timed_out, output_limited = _run_project(work, project_request)
        if stdout:
            sys.stdout.write(stdout)
            sys.stdout.flush()
        if timed_out:
            sys.stderr.write("CODER_TIMEOUT: 项目编译/运行超过时间上限。\n" + stderr)
            return 124
        if output_limited:
            sys.stderr.write("CODER_OUTPUT_LIMIT: 项目输出超过安全上限。\n" + stderr)
            return 125
        if stderr:
            sys.stderr.write(stderr + ("" if stderr.endswith("\n") else "\n"))
        return int(code)

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
    return int(pytest.main(["-q", "--disable-warnings", "--maxfail=5", *args.paths]))


if __name__ == "__main__":
    raise SystemExit(main())
