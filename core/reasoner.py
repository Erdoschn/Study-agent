import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .__debug__ import debug

@dataclass
class ReasoningDecision:
    action: str
    reasoning_summary: str

    tool: str | None = None
    arguments: dict[str, Any] | None = None

    goal: str = ""
    task_type: str = ""
    domain: str = ""

    claims: list[dict[str, Any]] | None = None

    finish_reason: str = ""
    model: str | None = None


class ModelClient(ABC):

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(ModelClient):

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
        }

        if json_mode:
            payload["response_format"] = {
                "type": "json_object"
            }

        request = urllib.request.Request(
            url=url,
            data=json.dumps(
                payload
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": (
                    f"Bearer {self.api_key}"
                ),
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw = response.read()

        except urllib.error.HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"LLM HTTP {exc.code}: {body}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"LLM 请求失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            data = json.loads(
                raw.decode("utf-8")
            )

            return data["choices"][0][
                "message"
            ]["content"]

        except Exception as exc:
            raise RuntimeError(
                f"LLM 返回格式异常：{exc}"
            ) from exc


class AgentReasoner:

    SYSTEM_PROMPT = """
你是 Study Agent 的核心决策器。

你不是最终回答器。
你的任务是分析当前 Agent 状态，并决定下一步行动。

可选 action：

SEARCH
CALCULATE
VERIFY
ANSWER
STOP

决策必须考虑：

1. 用户真正想解决什么问题
2. 当前已经知道什么
3. 当前还缺什么
4. 是否需要外部资料
5. 已经执行过哪些工具
6. 搜索结果是否足够
7. 是否需要验证
8. 是否应该继续行动

重要规则：

- 不要因为关键词出现就机械调用工具。
- 工具返回结果后必须重新分析。
- 信息不足时可以再次行动。
- 不确定时不要编造。
- 外部事实、最新信息、论文结论尽量获得证据。
- 数学推导不需要为了“有来源”而强行搜索。
- ANSWER 表示信息已经足够，可以交给 Teacher 生成最终答案。
- 不输出隐藏思维链，只输出简洁、可审计的 reasoning_summary。

必须只输出 JSON：

{
  "action": "SEARCH|CALCULATE|VERIFY|ANSWER|STOP",
  "reasoning_summary": "...",
  "tool": null,
  "arguments": {},
  "goal": "...",
  "task_type": "...",
  "domain": "...",
  "claims": [],
  "finish_reason": "..."
}

其中 reasoning_summary 是决策依据摘要，不是隐藏思维链。
"""

    def __init__(
        self,
        model_router,
        model_factory,
        allow_paid: bool = False,
    ):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def decide(self, state):

        with debug.scope(
            "AgentReasoner",
            f"DECIDE → step={state.step_count + 1}",
        ):

            prompt = self._build_prompt(
                state
            )

            debug.log(
                "AgentReasoner",
                f"PROMPT → {len(prompt)} chars",
            )

            excluded: set[str] = set()

            candidates = (
                self.model_router.select_candidates(
                    capability="reasoning",
                    allow_paid=self.allow_paid,
                    exclude=excluded,
                )
            )

            if not candidates:
                raise RuntimeError(
                    "没有可用于 Reasoning 的模型。"
                )

            errors = []

            for model in candidates:

                debug.log(
                    "AgentReasoner",
                    f"TRY MODEL → {model.name}",
                )

                try:
                    client = (
                        self.model_factory.create(
                            model
                        )
                    )

                    debug.log(
                        "AgentReasoner",
                        f"CALL LLM → {model.name}",
                    )

                    raw = client.generate(
                        self.SYSTEM_PROMPT,
                        prompt,
                        json_mode=True,
                    )

                    debug.log(
                        "AgentReasoner",
                        f"LLM RETURN → "
                        f"{len(raw)} chars",
                    )

                    decision = self._parse(
                        raw
                    )

                    self.model_router.registry.record_success(
                        model.name
                    )

                    decision.model = model.name

                    debug.log(
                        "AgentReasoner",
                        f"ACTION → "
                        f"{decision.action}",
                    )

                    if decision.tool:
                        debug.log(
                            "AgentReasoner",
                            f"TOOL → {decision.tool}",
                        )

                    return decision

                except Exception as exc:

                    self.model_router.registry.record_failure(
                        model.name
                    )

                    excluded.add(
                        model.name
                    )

                    debug.log(
                        "AgentReasoner",
                        f"MODEL FAILED → "
                        f"{model.name}",
                    )

                    errors.append(
                        f"{model.name}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

            raise RuntimeError(
                "所有 Reasoner 候选模型均调用失败：\n"
                + "\n".join(errors)
            )

    def _build_prompt(self, state):

        observations = []

        for step in state.steps:
            observations.append(
                {
                    "step": step.step_id,
                    "action": step.action,
                    "model": step.model,
                    "tool": step.tool,
                    "arguments": step.arguments,
                    "reasoning_summary": (
                        step.reasoning_summary
                    ),
                    "observation": step.observation,
                    "success": step.success,
                    "error": step.error,
                }
            )

        payload = {
            "question": state.question,
            "goal": state.goal,
            "task_type": state.task_type,
            "domain": state.domain,
            "student_state": {
                "known_topics": list(
                    state.student.known_topics
                ),
                "weak_topics": list(
                    state.student.weak_topics
                ),
                "misconceptions": (
                    state.student.misconceptions
                ),
            },
            "available_tools": [
                "search",
                "calculate",
                "verify",
            ],
            "previous_steps": observations,
            "evidence": state.evidence,
            "claims": state.claims,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _parse(raw):

        try:
            data = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Reasoner JSON 解析失败："
                f"{exc}\n原始输出：{raw}"
            ) from exc

        action = str(
            data.get(
                "action",
                "",
            )
        ).upper()

        valid_actions = {
            "SEARCH",
            "CALCULATE",
            "VERIFY",
            "ANSWER",
            "STOP",
        }

        if action not in valid_actions:
            raise RuntimeError(
                f"未知 action：{action}"
            )

        arguments = data.get(
            "arguments",
            {},
        )

        if not isinstance(
            arguments,
            dict,
        ):
            arguments = {}

        claims = data.get(
            "claims",
            [],
        )

        if not isinstance(
            claims,
            list,
        ):
            claims = []

        return ReasoningDecision(
            action=action,
            reasoning_summary=str(
                data.get(
                    "reasoning_summary",
                    "",
                )
            ),
            tool=(
                str(data["tool"])
                if data.get("tool") is not None
                else None
            ),
            arguments=arguments,
            goal=str(
                data.get(
                    "goal",
                    "",
                )
            ),
            task_type=str(
                data.get(
                    "task_type",
                    "",
                )
            ),
            domain=str(
                data.get(
                    "domain",
                    "",
                )
            ),
            claims=claims,
            finish_reason=str(
                data.get(
                    "finish_reason",
                    "",
                )
            ),
        )