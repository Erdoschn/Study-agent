from __future__ import annotations

import json
from typing import Any

from core.web_model import BrowserModel

from .state import CoderGoal, CoderState


class CoderReasoner:
    SYSTEM_PROMPT = """你是 StudyAgent 的 Python Coding Agent 决策器。
你不是普通聊天助手。你的唯一目标是让 Coding Goal 真正完成。
每轮只决定一个 action，然后观察 Harness 结果，再决定下一步。
不要在未完成验证前 FINISH；如果测试失败，分析错误并修改代码，再测试。

安全规则：
- 绝不能要求 shell、exec、eval、subprocess、os.system、powershell、cmd、bash 或任意 command。
- 文件只能通过 Harness 工具访问，不要假设可以直接读宿主机文件。
- 路径只能是 workspace 相对路径，禁止 ../、绝对路径和路径技巧。
- Python/pytest 执行由受限沙箱完成；不要要求联网执行 Python。
- 优先修改最小范围；先读代码，再 PATCH_FILE；大改才 WRITE_FILE。
- 修改代码后必须重新运行 pytest；过去通过的测试不能证明新修改仍然正确。
- 必须自己创建回归 pytest，不要为了通过测试修改测试去掩盖 bug。
- FINISH 只有 VERIFY_GOAL 返回 verified=true 后才允许。

action:
PLAN, SEARCH, LIST_FILES, READ_FILE, WRITE_FILE, PATCH_FILE, CREATE_TEST,
RUN_PYTHON, RUN_PYTEST, READ_DIFF, VERIFY_GOAL, FINISH

只输出 JSON：
{"action":"PLAN|SEARCH|LIST_FILES|READ_FILE|WRITE_FILE|PATCH_FILE|CREATE_TEST|RUN_PYTHON|RUN_PYTEST|READ_DIFF|VERIFY_GOAL|FINISH","arguments":{},"reasoning_summary":"","goal":{"description":"","required_files":[],"required_tests":[]},"answer":null}
"""

    def __init__(self, model: BrowserModel | None = None):
        self.model = model or BrowserModel(
            model="deepseek-web",
            user_data_dir=".coder-browser",
            cleanup_after_generate=True,
        )

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
            "last_test_result": state.last_test_result,
            "last_observation": str(state.last_observation)[:8000],
            "tools": tool_specs,
        }
        raw = self.model.generate(
            self.SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            json_mode=True,
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
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
            "PATCH_FILE", "CREATE_TEST", "RUN_PYTHON", "RUN_PYTEST",
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
