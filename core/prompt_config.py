from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.json"


def _resolve_path(path: str | Path | None = None) -> Path:
    value = path if path is not None else os.getenv(
        "STUDY_AGENT_PROMPTS",
        str(DEFAULT_PROMPTS_PATH),
    )
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=8)
def _load(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"找不到提示词配置文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"提示词配置格式错误：{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("提示词配置顶层必须是 JSON 对象。")
    return data


def load_prompts(path: str | Path | None = None) -> dict[str, Any]:
    return _load(str(_resolve_path(path)))


def get_prompt(name: str, path: str | Path | None = None) -> str:
    parts = [part for part in str(name or "").split(".") if part]
    if not parts:
        raise ValueError("提示词名称不能为空。")

    data: Any = load_prompts(path)
    for part in parts:
        if not isinstance(data, dict) or part not in data:
            raise KeyError(f"找不到提示词：{name}")
        data = data[part]

    if not isinstance(data, str) or not data.strip():
        raise ValueError(f"提示词必须是非空字符串：{name}")
    return data.strip()


def clear_prompt_cache() -> None:
    _load.cache_clear()
