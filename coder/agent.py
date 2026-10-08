from __future__ import annotations

import inspect
import os
import time
from threading import Event
from pathlib import Path

from core.__debug__ import debug

from core.cancellation import RunCancelled, raise_if_cancelled
from .filesystem import WorkspaceSecurityError
from .harness import CoderHarness
from .reasoner import CoderReasoner
from .state import CoderGoal, CoderState, CoderStep


DEFAULT_WORKSPACE = os.getenv("CODER_WORKSPACE", r"D:\Coder_workspace")
RUNTIME_ROOT = Path(__file__).resolve().parents[1]


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
        reuse_chat: bool = True,
        min_send_interval_seconds: float = 5.0,
        close_model_on_run: bool = True,
        cancellation_event: Event | None = None,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        self._validate_workspace_boundary()
        self.debug_mode = bool(debug_mode)
        self.close_model_on_run = bool(close_model_on_run)
        self.cancellation_event = cancellation_event
        if search_router is None:
            from tools.search import ArxivSearchProvider, SearchRouter, WikipediaSearchProvider
            search_router = SearchRouter()
            search_router.register(ArxivSearchProvider())
            search_router.register(WikipediaSearchProvider())
        self.reasoner = reasoner
        self.harness = harness or CoderHarness(
            str(self.workspace),
            search_router=search_router,
            sandbox=sandbox,
            cancellation_event=cancellation_event,
        )
        if cancellation_event is not None:
            try:
                self.harness.cancellation_event = cancellation_event
            except Exception:
                pass
        if self.reasoner is None:
            reasoner_kwargs = {
                "debug_mode": self.debug_mode,
                "reuse_chat": reuse_chat,
                "min_send_interval_seconds": min_send_interval_seconds,
            }
            if cancellation_event is not None:
                reasoner_kwargs["cancellation_event"] = cancellation_event
            self.reasoner = CoderReasoner(**reasoner_kwargs)
        self.max_runtime_seconds = max(30.0, float(max_runtime_seconds))

    def _validate_workspace_boundary(self) -> None:
        """Prevent Coder from operating on or above its own runtime source tree."""
        try:
            self.workspace.relative_to(RUNTIME_ROOT)
        except ValueError:
            try:
                RUNTIME_ROOT.relative_to(self.workspace)
            except ValueError:
                return
        raise WorkspaceSecurityError(
            "Coder workspace 不能覆盖 Study-agent 自身源码目录；"
            "请使用独立的目标项目 workspace。"
        )

    def run(self, request: str, *, event_hook=None) -> CoderState:
        request = str(request or "").strip()
        if not request:
            raise ValueError("Coder 请求不能为空。")
        state = CoderState(request=request, project=self.workspace.name)

        def emit(event: dict) -> None:
            if callable(event_hook):
                try:
                    event_hook(event)
                except Exception:
                    pass

        emit({"type": "started", "request": request})
        try:
            raise_if_cancelled(self.cancellation_event)
            preflight = getattr(self.harness.sandbox, "preflight", None)
            if callable(preflight):
                if self.cancellation_event is None:
                    preflight()
                else:
                    try:
                        parameters = inspect.signature(preflight).parameters
                    except (TypeError, ValueError):
                        parameters = {}
                    accepts_cancel = (
                        "cancellation_event" in parameters
                        or any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters.values()
                        )
                    )
                    if accepts_cancel:
                        preflight(cancellation_event=self.cancellation_event)
                    else:
                        preflight()
                    raise_if_cancelled(self.cancellation_event)
            initial = getattr(self.harness.backup, "ensure_initial_snapshot", None)
            if callable(initial):
                snapshot = initial(0)
                state.initial_backup_generation = snapshot.generation
        except RunCancelled:
            state.finished = True
            state.goal_verified = False
            state.cancelled = True
            state.summary = "任务已被用户中止。"
            state.metrics["cancelled"] = True
            emit({"type": "cancelled", "state": state})
            return state
        except Exception as exc:
            if self.cancellation_event is not None and self.cancellation_event.is_set():
                state.finished = True
                state.cancelled = True
                state.summary = "任务已被用户中止。"
                state.metrics["cancelled"] = True
                emit({"type": "cancelled", "state": state})
                return state
            state.error = f"Coder 启动安全检查失败：{type(exc).__name__}: {exc}"
            debug.log("CoderAgent", state.error)
            emit({"type": "error", "state": state})
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
                raise_if_cancelled(self.cancellation_event)
                if time.monotonic() - started >= self.max_runtime_seconds:
                    raise TimeoutError("Coder 安全运行时间上限已到，已拒绝继续执行。")

                decision = self.reasoner.decide(state, self.harness.tool_specs())
                raise_if_cancelled(self.cancellation_event)
                action = decision["action"]
                arguments = decision["arguments"]

                if action == "PLAN":
                    self._apply_plan(state, decision.get("goal", {}))
                    observation = {"status": "PLAN_SET", "goal": state.goal.__dict__}
                    state.add_step(CoderStep(state.step_count + 1, action, arguments, observation))
                    emit({"type": "step", "step": state.steps[-1]})
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
                        emit({"type": "step", "step": state.steps[-1]})
                    except RunCancelled:
                        raise
                    except Exception as exc:
                        state.add_step(CoderStep(
                            state.step_count + 1, action, arguments,
                            {"error": f"{type(exc).__name__}: {exc}"},
                            success=False,
                            error=str(exc),
                        ))
                        emit({"type": "step", "step": state.steps[-1]})
                    continue

                if action == "FINISH":
                    observation = self.harness.execute("VERIFY_GOAL", {}, state)
                    ok = bool(observation.get("verified"))
                    state.add_step(CoderStep(
                        state.step_count + 1, action, arguments, observation, success=ok,
                        error="" if ok else "Goal 未满足，继续 Agent Loop。",
                    ))
                    emit({"type": "step", "step": state.steps[-1]})
                    if ok:
                        state.goal_verified = True
                        state.finished = True
                        state.summary = self._build_completion_summary(state)
                        break
                    continue

                try:
                    raise_if_cancelled(self.cancellation_event)
                    observation = self.harness.execute(action, arguments, state)
                    state.add_step(CoderStep(state.step_count + 1, action, arguments, observation))
                    emit({"type": "step", "step": state.steps[-1]})
                except RunCancelled:
                    raise
                except Exception as exc:
                    if self.cancellation_event is not None and self.cancellation_event.is_set():
                        raise RunCancelled("Coder 任务已被用户中止。") from exc
                    observation = {"error": f"{type(exc).__name__}: {exc}"}
                    state.add_step(CoderStep(
                        state.step_count + 1, action, arguments, observation,
                        success=False, error=str(exc),
                    ))
                    emit({"type": "step", "step": state.steps[-1]})
            state.metrics["steps"] = state.step_count
            state.metrics["modified_files"] = len(state.modified_files)
            state.metrics["chat_resets"] = state.chat_resets
            state.metrics["test_runs"] = sum(
                1 for step in state.steps if step.action in {"RUN_PYTHON", "RUN_PYTEST"}
            )
            emit({"type": "finished", "state": state})
            return state
        except RunCancelled as exc:
            state.finished = True
            state.goal_verified = False
            state.cancelled = True
            state.error = None
            state.metrics["cancelled"] = True
            state.summary = "任务已被用户中止。"
            emit({"type": "cancelled", "state": state})
            return state
        except Exception as exc:
            if self.cancellation_event is not None and self.cancellation_event.is_set():
                state.finished = True
                state.goal_verified = False
                state.cancelled = True
                state.error = None
                state.metrics["cancelled"] = True
                state.summary = "任务已被用户中止。"
                emit({"type": "cancelled", "state": state})
                return state
            state.error = f"Coder 执行失败：{type(exc).__name__}: {exc}"
            debug.log("CoderAgent", state.error)
            emit({"type": "error", "state": state})
            return state
        finally:
            if self.close_model_on_run:
                model = getattr(self.reasoner, "model", None)
                close = getattr(model, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    @staticmethod
    def _build_completion_summary(state: CoderState) -> str:
        """Build a deterministic user-facing summary without another model call."""
        change_lines: list[str] = []
        seen_changes: set[tuple[str, str]] = set()
        action_labels = {
            "PATCH_FILE": "修改",
            "WRITE_FILE": "写入/更新",
            "WRITE_NOTEBOOK": "写入/更新 Notebook",
            "CREATE_TEST": "新增测试",
        }
        for step in state.steps:
            label = action_labels.get(step.action)
            if not label or not isinstance(step.arguments, dict):
                continue
            path = str(step.arguments.get("path", "")).strip()
            if not path:
                continue
            key = (label, path)
            if key in seen_changes:
                continue
            seen_changes.add(key)
            change_lines.append(f"{label} {path}")

        lines = ["任务已完成，并通过 Goal 验证。"]
        if change_lines:
            lines.append("本次修改：")
            lines.extend(f"  - {item}" for item in change_lines)
        elif state.modified_files:
            lines.append("本次修改文件：" + ", ".join(sorted(state.modified_files)))

        test = state.last_test_result
        if isinstance(test, dict):
            kind = str(test.get("kind", "test"))
            if test.get("passed") is True:
                lines.append(f"验证：{kind} 测试通过。")
            elif test.get("passed") is False:
                lines.append(f"验证：{kind} 测试未通过。")

        if state.chat_resets:
            lines.append(f"会话重置：{state.chat_resets} 次。")
        return "\n".join(lines)

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
