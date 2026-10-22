"""Private, path-safe Markdown storage for generated study questions."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage_security import ensure_private_directory
from .user_identity import normalize_username

_FILENAME_RE = re.compile(r"^(?:question|answer)-\d{8}_\d{6}-[0-9a-f]{12}\.md$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_STATE_BYTES = 64 * 1024


def output_root(root: str | Path | None = None) -> Path:
    """Return the repository-root question directory with private permissions."""
    destination = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "出题"
    return ensure_private_directory(destination)


def profile_token(username: object) -> str:
    canonical = normalize_username(username)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _profile_directory(username: object, root: str | Path | None = None) -> tuple[str, Path]:
    token = profile_token(username)
    base = output_root(root)
    return token, ensure_private_directory(base / token)


def _safe_filename(value: object) -> str:
    filename = str(value or "")
    if not _FILENAME_RE.fullmatch(filename):
        raise ValueError("题目文件名格式无效。")
    return filename


def _atomic_write(path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有文件：{path.name}")
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".pending-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def save_assessment_files(
    assessment: dict[str, Any],
    username: object,
    *,
    root: str | Path | None = None,
) -> dict[str, str]:
    """Persist the problem and answer separately, rolling back partial pairs."""
    if not isinstance(assessment, dict):
        raise TypeError("assessment must be a dictionary")
    question = str(assessment.get("question") or "").strip()
    answer = str(assessment.get("expected_answer") or "").strip()
    topic = str(assessment.get("primary_concept") or "").strip()
    if not question or not answer or not topic:
        raise ValueError("出题结果缺少题目、参考答案或考查知识点。")
    rubric_raw = assessment.get("rubric", [])
    rubric = (
        [str(item).strip() for item in rubric_raw if str(item).strip()]
        if isinstance(rubric_raw, list)
        else []
    )
    token, directory = _profile_directory(username, root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nonce = uuid4().hex[:12]
    question_name = f"question-{stamp}-{nonce}.md"
    answer_name = f"answer-{stamp}-{nonce}.md"
    question_path = directory / question_name
    answer_path = directory / answer_name

    question_document = (
        f"# 练习题\n\n"
        f"- 考查知识点：{topic}\n"
        f"- 难度：{str(assessment.get('difficulty_level') or assessment.get('difficulty') or 'graduate')}\n"
        f"- 题型：{str(assessment.get('question_type') or 'open_ended')}\n\n"
        f"## 题目\n\n{question}\n"
    )
    answer_document = (
        f"# 参考答案与评分标准\n\n"
        f"- 考查知识点：{topic}\n"
        f"- 对应题目：[{question_name}]({question_name})\n\n"
        f"## 参考解答\n\n{answer}\n\n"
        f"## 评分要点\n\n"
        + ("\n".join(f"- {item}" for item in rubric) if rubric else "- 请根据参考解答核对关键推理步骤。")
        + "\n"
    )
    written: list[Path] = []
    try:
        _atomic_write(question_path, question_document)
        written.append(question_path)
        _atomic_write(answer_path, answer_document)
        written.append(answer_path)
    except Exception:
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return {
        "user_token": token,
        "question_file": question_name,
        "answer_file": answer_name,
        "question_path": f"出题/{token}/{question_name}",
        "answer_path": f"出题/{token}/{answer_name}",
        "question_url_path": f"/v1/assessment/files/{token}/{question_name}",
        "answer_url_path": f"/v1/assessment/files/{token}/{answer_name}",
        "directory": str(directory),
    }


def read_assessment_file(
    user_token: object,
    filename: object,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve only a generated Markdown filename inside its profile directory."""
    token = str(user_token or "")
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError("题目文件 Profile 标识无效。")
    safe_name = _safe_filename(filename)
    base = output_root(root)
    profile = ensure_private_directory(base / token)
    candidate = profile / safe_name
    try:
        status = candidate.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError("找不到该 Markdown 文件。") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise PermissionError("拒绝读取非普通文件或符号链接。")
    if status.st_size > 8 * 1024 * 1024:
        raise ValueError("Markdown 文件超过安全读取上限。")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != profile.resolve(strict=True):
        raise PermissionError("Markdown 路径超出允许目录。")
    return resolved


def _state_path(username: object, task_id: object, root: str | Path | None) -> tuple[str, Path]:
    token, directory = _profile_directory(username, root)
    task = str(task_id or "").strip()
    if not task or len(task) > 128 or any(ord(char) < 32 for char in task):
        raise ValueError("task_id 格式无效。")
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
    return token, directory / f"state-{digest}.json"


def load_question_generation_state(
    username: object,
    task_id: object,
    *,
    root: str | Path | None = None,
) -> dict[str, Any] | None:
    _token, path = _state_path(username, task_id, root)
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise PermissionError("出题状态文件必须是普通文件。")
    if status.st_size > _MAX_STATE_BYTES:
        raise ValueError("出题状态文件超过安全读取上限。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("出题状态文件格式无效。")
    count = payload.get("completed_turn_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("出题状态中的对话轮数无效。")
    _safe_filename(payload.get("question_file"))
    _safe_filename(payload.get("answer_file"))
    return payload


def save_question_generation_state(
    username: object,
    task_id: object,
    state: dict[str, Any],
    *,
    root: str | Path | None = None,
) -> None:
    _token, path = _state_path(username, task_id, root)
    payload = dict(state)
    payload["schema_version"] = 1
    count = payload.get("completed_turn_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("completed_turn_count 必须为正整数。")
    _safe_filename(payload.get("question_file"))
    _safe_filename(payload.get("answer_file"))
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_STATE_BYTES:
        raise ValueError("出题状态超过安全大小上限。")
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".state-pending-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
