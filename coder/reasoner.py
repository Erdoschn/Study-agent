from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from core.web_model import BrowserModel
from core.prompt_config import get_prompt

from .state import CoderGoal, CoderState


class CoderReasoner:
    SYSTEM_PROMPT = get_prompt("coder.reasoner")


    def __init__(
        self,
        model: BrowserModel | None = None,
        *,
        debug_mode: bool = False,
        reuse_chat: bool = True,
        min_send_interval_seconds: float = 5.0,
    ):
        self.model = model or BrowserModel(
            model="deepseek-web",
            user_data_dir=".coder-browser",
            cleanup_after_generate=False,
            reuse_chat=reuse_chat,
            min_send_interval_seconds=min_send_interval_seconds,
            debug_mode=debug_mode,
        )

    def new_chat(self) -> None:
        """Reset only the Coder's browser conversation when needed."""
        reset = getattr(self.model, "new_chat", None)
        if not callable(reset):
            raise RuntimeError("当前 Coder WebModel 不支持新会话。")
        reset()

    def decide(self, state: CoderState, tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "request": state.request,
            "goal": (
                state.goal.__dict__ if state.goal else {
                    "description": state.request,
                    "required_files": [],
                    "required_tests": [],
                    "must_modify": True,
                    "must_create_tests": True,
                    "must_pass_tests": True,
                }
            ),
            "step": state.step_count,
            "modified_files": sorted(state.modified_files),
            "created_tests": sorted(state.created_tests),
            "last_test_result_untrusted": self._untrusted(state.last_test_result),
            "last_observation_untrusted": self._untrusted(state.last_observation),
            "chat": {
                "reuse_enabled": bool(getattr(self.model, "reuse_chat", False)),
                "reset_count": int(getattr(state, "chat_resets", 0)),
            },
            "tools": tool_specs,
        }
        raw = self.model.generate(
            self.SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            json_mode=True,
        )
        return self._parse(raw)

    @staticmethod
    def _scrub_text(value: str, *, redact_paths: bool = True) -> str:
        scrubbed = "".join(
            ch for ch in str(value or "")
            if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf"}
        )
        if not redact_paths:
            return scrubbed
        scrubbed = re.sub(
            r"(?i)(?:[A-Z]:[\\/]|\\\\)[^\s\"'<>]+",
            "<REDACTED_HOST_PATH>",
            scrubbed,
        )
        scrubbed = re.sub(
            r"(?i)(?:\.coder-backup[\\/])[^\s\"'<>]+",
            "<REDACTED_BACKUP_PATH>",
            scrubbed,
        )
        return re.sub(
            r"(?<![A-Za-z0-9_])/(?:workspace|host_mnt|run/desktop/mnt/host)(?:/[^\s\"'<>]*)?",
            "<REDACTED_SANDBOX_PATH>",
            scrubbed,
        )

    @classmethod
    def _untrusted(cls, value: Any) -> str:
        if value is None:
            text = "<empty>"
        else:
            try:
                if isinstance(value, dict):
                    safe = {}
                    for key, item in value.items():
                        if str(key).casefold() == "content":
                            safe[key] = cls._scrub_text(str(item), redact_paths=False)
                        else:
                            safe[key] = item
                    text = json.dumps(safe, ensure_ascii=False, default=str)
                else:
                    text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)

        if isinstance(value, dict):
            # The READ_FILE content itself is source material, not host-path
            # metadata. Preserve it so a coding model can actually inspect and
            # edit the file; continue sanitizing all non-content fields.
            try:
                parts = []
                for key, item in value.items():
                    if str(key).casefold() == "content":
                        parts.append(
                            json.dumps(
                                {key: cls._scrub_text(str(item), redact_paths=False)},
                                ensure_ascii=False,
                            )
                        )
                    else:
                        parts.append(
                            json.dumps(
                                {key: item},
                                ensure_ascii=False,
                                default=str,
                            )
                        )
                if parts:
                    text = " ".join(parts)
            except Exception:
                pass

        scrubbed = cls._scrub_text(text, redact_paths=False)
        # Re-serialize the non-content representation through a safe pass so
        # host/sandbox path redaction still applies outside source content.
        if isinstance(value, dict):
            try:
                rendered = []
                for key, item in value.items():
                    key_text = cls._scrub_text(str(key), redact_paths=True)
                    if str(key).casefold() == "content":
                        value_text = cls._scrub_text(str(item), redact_paths=False)
                    else:
                        value_text = cls._scrub_text(
                            json.dumps(item, ensure_ascii=False, default=str),
                            redact_paths=True,
                        )
                    rendered.append(f"{key_text}={value_text}")
                scrubbed = "{" + ", ".join(rendered) + "}"
            except Exception:
                pass
        else:
            scrubbed = cls._scrub_text(text, redact_paths=True)

        return (
            "<UNTRUSTED_TOOL_OUTPUT>\n"
            + scrubbed[:12000]
            + "\n</UNTRUSTED_TOOL_OUTPUT>"
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if len(text.encode("utf-8")) > 65_536:
            raise RuntimeError("CoderReasoner 输出超过安全长度上限。")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise RuntimeError("CoderReasoner JSON 解析失败。")
            data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise RuntimeError("CoderReasoner JSON 顶层必须是对象。")
        action = str(data.get("action", "")).upper()
        allowed = {
            "PLAN", "SEARCH", "LIST_FILES", "READ_FILE", "WRITE_FILE",
            "WRITE_NOTEBOOK", "PATCH_FILE", "CREATE_TEST", "RUN_PYTHON", "RUN_PYTEST",
            "NEW_CHAT",
            "READ_DIFF", "VERIFY_GOAL", "FINISH",
        }
        if action not in allowed:
            raise RuntimeError(f"CoderReasoner 未知 action：{action}")
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            raise RuntimeError("CoderReasoner arguments 必须是对象。")
        return {
            "action": action,
            "arguments": args,
            "reasoning_summary": str(data.get("reasoning_summary", ""))[:500],
            "goal": data.get("goal", {}) if isinstance(data.get("goal", {}), dict) else {},
            "answer": str(data.get("answer", "")).strip(),
        }
