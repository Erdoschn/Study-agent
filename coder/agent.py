from __future__ import annotations

import inspect
import os
import re
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

    @staticmethod
    def _request_requires_tests(request: str) -> bool:
        return bool(re.search(
            r"(?i)\btests?\b|\bpytest\b|测试|回归|单元测试",
            str(request or ""),
        ))

    @staticmethod
    def _request_requires_test_creation(request: str) -> bool:
        text = str(request or "").casefold()
        return any(token in text for token in (
            "create tests", "add tests", "write tests", "新增测试", "增加测试",
            "补充测试", "创建测试", "编写测试",
        ))

    def _workspace_has_tests(self) -> bool:
        list_files = getattr(getattr(self.harness, "fs", None), "list_files", None)
        if not callable(list_files):
            return False
        try:
            paths = list_files()
        except Exception:
            return False
        for item in paths:
            path = str(item).replace("\\", "/").casefold()
            if path.endswith(".py") and (
                "/tests/" in "/" + path or Path(path).name.startswith("test_")
            ):
                return True
        return False

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

    def run(self, request: str, *, event_hook=None, user_interaction=None) -> CoderState:
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

        existing_tests = self._workspace_has_tests()
        explicitly_requests_tests = self._request_requires_tests(request)
        # The planner decides which files/tests are needed. The user's original
        # request remains immutable in state.request and is carried to every turn.
        state.goal = CoderGoal(
            description=request,
            must_modify=True,
            must_create_tests=False,
            must_pass_tests=existing_tests or explicitly_requests_tests,
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

                # No tool may run before the first planning decision. Keep
                # returning the invariant to the model instead of executing an
                # opportunistic first action.
                if not state.plan_confirmed and action not in {"PLAN", "STOP"}:
                    message = (
                        "PLAN_REQUIRED：第一步必须规划长期目标、成功标准、里程碑和初始文件范围；"
                        "当前动作未执行。请先返回 PLAN。"
                    )
                    state.add_step(CoderStep(
                        state.step_count + 1,
                        "PLAN_REQUIRED",
                        {},
                        {"error": message, "attempted_action": action},
                        success=False,
                        error=message,
                    ))
                    emit({"type": "step", "step": state.steps[-1]})
                    continue

                if action == "STOP":
                    reason = (
                        str(decision.get("reasoning_summary", "")).strip()
                        or str(decision.get("answer", "")).strip()
                        or "模型无法给出有效的结构化操作，已安全停止。"
                    )
                    state.finished = True
                    state.goal_verified = False
                    state.error = "Coder 安全停止：" + reason
                    state.summary = state.error
                    state.add_step(CoderStep(
                        state.step_count + 1,
                        action,
                        arguments,
                        {"status": "stopped", "reason": reason},
                        success=False,
                        error=reason,
                    ))
                    emit({"type": "step", "step": state.steps[-1]})
                    emit({"type": "error", "state": state})
                    return state

                if action == "ASK_USER":
                    question = str(
                        arguments.get("question", "")
                        or decision.get("reasoning_summary", "")
                    ).strip()[:2000]
                    if not question:
                        question = "Coder 遇到阻塞。请提供建议，或等待 10 秒由 Coder 自主规划下一步。"
                    normalized_question = " ".join(question.casefold().split())
                    if normalized_question in {
                        " ".join(item.casefold().split())
                        for item in state.asked_user_questions
                    }:
                        answer = None
                        observation = {
                            "status": "already_asked",
                            "question": question,
                            "response": None,
                            "next": "该问题之前已询问且无回复；不要重复询问，继续自主规划。",
                        }
                    else:
                        state.asked_user_questions.append(question)
                        answer = user_interaction(question) if callable(user_interaction) else None
                        if answer and str(answer).strip():
                            answer = str(answer).strip()[:12_000]
                            state.user_responses.append({
                                "question": question,
                                "response": answer,
                            })
                            observation = {
                                "status": "answered",
                                "question": question,
                                "response": answer,
                                "next": "将用户回应纳入原始目标约束，重新规划下一步。",
                            }
                        else:
                            observation = {
                                "status": "timeout",
                                "question": question,
                                "response": None,
                                "next": "用户未在 10 秒内点击回应按钮；自主规划下一步，不要重复询问同一问题。",
                            }
                    state.add_step(CoderStep(
                        state.step_count + 1, action, arguments, observation,
                        success=bool(answer),
                    ))
                    emit({"type": "step", "step": state.steps[-1]})
                    continue

                if action == "PLAN":
                    self._apply_plan(state, decision.get("goal", {}))
                    observation = {
                        "status": "PLAN_SET",
                        "long_term_goal": state.request,
                        "goal": state.goal.__dict__,
                        "current_milestone": state.current_milestone,
                        "completed_milestones": state.completed_milestones,
                    }
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
                    message = str(exc)
                    if any(marker in message for marker in (
                        "不在当前计划范围",
                        "必须先完成 PLAN",
                        "必须先执行 PLAN",
                    )):
                        observation = {"error": f"{type(exc).__name__}: {exc}"}
                        state.add_step(CoderStep(
                            state.step_count + 1, action, arguments, observation,
                            success=False, error=message,
                        ))
                        emit({"type": "step", "step": state.steps[-1]})
                        scope_failures = int(state.metrics.get("scope_write_failures", 0)) + 1
                        state.metrics["scope_write_failures"] = scope_failures
                        # One scope rejection should trigger a PLAN update. If
                        # the model ignores it twice, offer the user a 10-second
                        # response window; timeout means autonomous replanning.
                        if scope_failures == 2 and callable(user_interaction):
                            path = str(arguments.get("path", "")).strip()
                            question = (
                                f"Coder 两次尝试操作计划范围外的文件 {path or '(未指定路径)'}。"
                                "如果确有必要，请说明应扩展到哪些文件；10 秒内点击“回应”可输入，"
                                "否则 Coder 会根据原始目标自行重规划。"
                            )
                            answer = user_interaction(question)
                            if answer and str(answer).strip():
                                state.user_responses.append({
                                    "question": question,
                                    "response": str(answer).strip()[:12_000],
                                })
                            state.add_step(CoderStep(
                                state.step_count + 1,
                                "ASK_USER",
                                {"question": question},
                                {
                                    "status": "answered" if answer else "timeout",
                                    "response": str(answer).strip()[:12_000] if answer else None,
                                    "next": "请根据原始目标和用户回应/超时结果重新规划文件范围。",
                                },
                                success=bool(answer),
                            ))
                            emit({"type": "step", "step": state.steps[-1]})
                        continue
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
        """Apply/extend the explicit task plan without mutating the user's request."""
        goal = state.goal or CoderGoal(state.request)
        if not isinstance(raw, dict):
            raw = {}

        def merge_strings(existing, incoming, limit=40):
            result = list(existing or [])
            seen = {str(item).replace("\\", "/").casefold() for item in result}
            if isinstance(incoming, list):
                for item in incoming:
                    value = str(item or "").strip()
                    if not value:
                        continue
                    key = value.replace("\\", "/").casefold()
                    if key not in seen:
                        result.append(value)
                        seen.add(key)
                    if len(result) >= limit:
                        break
            return result[:limit]

        goal.required_files = merge_strings(goal.required_files, raw.get("required_files", []))
        goal.required_tests = merge_strings(goal.required_tests, raw.get("required_tests", []))
        goal.scope_files = merge_strings(
            goal.scope_files,
            raw.get("scope_files", raw.get("allowed_files", [])),
        )
        new_milestones = raw.get("milestones", [])
        if isinstance(new_milestones, list) and new_milestones:
            goal.milestones = merge_strings(goal.milestones, new_milestones, limit=20)
        new_criteria = raw.get("success_criteria", [])
        if isinstance(new_criteria, list) and new_criteria:
            goal.success_criteria = merge_strings(goal.success_criteria, new_criteria, limit=20)

        completed = raw.get("completed_milestones", [])
        state.completed_milestones = merge_strings(
            state.completed_milestones,
            completed,
            limit=20,
        )
        description = str(raw.get("description", "")).strip()
        if description:
            goal.description = description[:2000]

        # The planner may choose whether dedicated tests are needed. Existing
        # test-suite requirements inferred by the agent cannot be weakened.
        if isinstance(raw.get("must_create_tests"), bool):
            goal.must_create_tests = raw["must_create_tests"]
        goal.must_pass_tests = goal.must_pass_tests or raw.get("must_pass_tests") is True
        goal.must_modify = True
        state.goal = goal
        state.plan_confirmed = True
        state.current_milestone = next(
            (item for item in goal.milestones if item not in state.completed_milestones),
            "",
        )
