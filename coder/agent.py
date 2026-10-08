from __future__ import annotations

import os
import time
from pathlib import Path

from core.__debug__ import debug

from .harness import CoderHarness
from .reasoner import CoderReasoner
from .state import CoderGoal, CoderState, CoderStep


DEFAULT_WORKSPACE = os.getenv("CODER_WORKSPACE", r"D:\Coder_workspace")


class CoderAgent:
    """Autonomous Python coding agent with a fail-closed Harness."""

    def __init__(
        self,
        workspace: str | Path = DEFAULT_WORKSPACE,
        *,
        reasoner=None,
        harness=None,
        search_router=None,
        sandbox=None,
        max_runtime_seconds: float = 1800,
        debug_mode: bool = False,
    ):
        self.workspace = Path(workspace)
        self.debug_mode = bool(debug_mode)
        if search_router is None:
            from tools.search import ArxivSearchProvider, SearchRouter, WikipediaSearchProvider
            search_router = SearchRouter()
            search_router.register(ArxivSearchProvider())
            search_router.register(WikipediaSearchProvider())
        self.reasoner = reasoner
        self.harness = harness or CoderHarness(str(self.workspace), search_router=search_router, sandbox=sandbox)
        if self.reasoner is None:
            self.reasoner = CoderReasoner(debug_mode=self.debug_mode)
        self.max_runtime_seconds = max(30.0, float(max_runtime_seconds))

    def run(self, request: str) -> CoderState:
        request = str(request or "").strip()
        if not request:
            raise ValueError("Coder 请求不能为空。")
        state = CoderState(request=request)
        try:
            preflight = getattr(self.harness.sandbox, "preflight", None)
            if callable(preflight):
                preflight()
            initial = getattr(self.harness.backup, "ensure_initial_snapshot", None)
            if callable(initial):
                snapshot = initial(0)
                state.initial_backup_generation = snapshot.generation
        except Exception as exc:
            state.error = f"Coder 启动安全检查失败：{type(exc).__name__}: {exc}"
            debug.log("CoderAgent", state.error)
            return state

        state.goal = CoderGoal(
            description=request,
            must_modify=True,
            must_create_tests=True,
            must_pass_tests=True,
        )
        started = time.monotonic()
        try:
            while not state.goal_verified:
                if time.monotonic() - started >= self.max_runtime_seconds:
                    raise TimeoutError("Coder 安全运行时间上限已到，已拒绝继续执行。")

                decision = self.reasoner.decide(state, self.harness.tool_specs())
                action = decision["action"]
                arguments = decision["arguments"]

                if action == "PLAN":
                    self._apply_plan(state, decision.get("goal", {}))
                    observation = {"status": "PLAN_SET", "goal": state.goal.__dict__}
                    state.add_step(CoderStep(state.step_count + 1, action, arguments, observation))
                    continue

                if action == "NEW_CHAT":
                    new_chat = getattr(self.reasoner, "new_chat", None)
                    if not callable(new_chat):
                        raise RuntimeError("当前 Coder Reasoner 不支持 NEW_CHAT。")
                    try:
                        new_chat()
                        state.chat_resets += 1
                        observation = {
                            "status": "NEW_CHAT",
                            "reset_count": state.chat_resets,
                        }
                        state.add_step(CoderStep(
                            state.step_count + 1, action, arguments, observation,
                        ))
                    except Exception as exc:
                        state.add_step(CoderStep(
                            state.step_count + 1, action, arguments,
                            {"error": f"{type(exc).__name__}: {exc}"},
                            success=False,
                            error=str(exc),
                        ))
                    continue

                if action == "FINISH":
                    observation = self.harness.execute("VERIFY_GOAL", {}, state)
                    ok = bool(observation.get("verified"))
                    state.add_step(CoderStep(
                        state.step_count + 1, action, arguments, observation, success=ok,
                        error="" if ok else "Goal 未满足，继续 Agent Loop。",
                    ))
                    if ok:
                        state.finished = True
                        break
                    continue

                try:
                    observation = self.harness.execute(action, arguments, state)
                    state.add_step(CoderStep(state.step_count + 1, action, arguments, observation))
                except Exception as exc:
                    observation = {"error": f"{type(exc).__name__}: {exc}"}
                    state.add_step(CoderStep(
                        state.step_count + 1, action, arguments, observation,
                        success=False, error=str(exc),
                    ))
            state.metrics["steps"] = state.step_count
            state.metrics["modified_files"] = len(state.modified_files)
            state.metrics["chat_resets"] = state.chat_resets
            state.metrics["test_runs"] = sum(
                1 for step in state.steps if step.action in {"RUN_PYTHON", "RUN_PYTEST"}
            )
            return state
        except Exception as exc:
            state.error = f"Coder 执行失败：{type(exc).__name__}: {exc}"
            debug.log("CoderAgent", state.error)
            return state
        finally:
            model = getattr(self.reasoner, "model", None)
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @staticmethod
    def _apply_plan(state: CoderState, raw: dict) -> None:
        goal = state.goal or CoderGoal(state.request)
        files = raw.get("required_files", [])
        tests = raw.get("required_tests", [])
        if isinstance(files, list):
            goal.required_files = [str(x).strip() for x in files if str(x).strip()][:20]
        if isinstance(tests, list):
            goal.required_tests = [str(x).strip() for x in tests if str(x).strip()][:20]
        goal.description = str(raw.get("description", goal.description)).strip() or goal.description
        # Security policy: a model cannot weaken mandatory coding verification.
        goal.must_modify = True
        goal.must_create_tests = True
        goal.must_pass_tests = True
        state.goal = goal
