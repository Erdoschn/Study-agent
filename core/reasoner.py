import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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


class ModelClient(ABC):
    """
    LLM 接口。

    Agent 不关心底层究竟是：
    - OpenAI
    - Gemini
    - DeepSeek
    - Ollama
    - 其他 OpenAI-compatible API
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(ModelClient):
    """
    通用 OpenAI-compatible API 客户端。

    可用于：
    - OpenAI-compatible 服务
    - 本地模型服务
    - Ollama OpenAI-compatible endpoint
    - 其他兼容接口
    """

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
                "type": "json_object",
            }

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
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

            return data["choices"][0]["message"]["content"]

        except Exception as exc:
            raise RuntimeError(
                f"LLM 返回格式异常：{exc}"
            ) from exc


class AgentReasoner:
    """
    Agent 的核心推理器。

    输入：
        当前 AgentState

    输出：
        一个结构化行动决策。

    注意：
        模型只提供“决策摘要”。
        不把隐藏内部思维链暴露给最终用户。
    """

    SYSTEM_PROMPT = """
你是 Study Agent 的核心决策器。

你的任务不是直接回答用户，而是决定 Agent 下一步应该做什么。

你可以选择：

SEARCH
    使用搜索工具获取外部资料。

CALCULATE
    使用计算工具。

VERIFY
    验证已有关键结论或 Claim。

ANSWER
    当前信息已经足够，可以生成最终回答。

STOP
    当前无法继续执行。

你的决策必须基于：
1. 用户真正的问题和目标
2. 当前已经掌握的信息
3. 已经执行过的工具
4. 当前仍然缺少的信息
5. 是否需要外部证据
6. 是否需要验证关键结论

重要规则：
- 不要因为出现某个关键词就机械搜索。
- 可以多轮行动。
- 工具返回结果后必须重新判断。
- 不确定时优先获取证据，而不是编造。
- 数学推导可以直接推理，不需要强行搜索。
- 外部事实、最新信息、论文结论等应尽量获得来源。
- 不得把未经验证的推测当成事实。
- VERIFY 用于重要 Claim 的验证。
- ANSWER 只能在信息足够时使用。

只输出 JSON，不要输出 Markdown。

JSON 格式：

{
  "action": "SEARCH|CALCULATE|VERIFY|ANSWER|STOP",
  "reasoning_summary": "对当前决策的简洁依据",
  "tool": "工具名，没有工具时为 null",
  "arguments": {},
  "goal": "用户真正目标",
  "task_type": "任务类型",
  "domain": "问题领域",
  "claims": [],
  "finish_reason": "只有 ANSWER/STOP 时填写"
}
"""

    def __init__(self, client: ModelClient):
        self.client = client

    def decide(self, state) -> ReasoningDecision:
        prompt = self._build_prompt(state)

        raw = self.client.generate(
            self.SYSTEM_PROMPT,
            prompt,
            json_mode=True,
        )

        return self._parse(raw)

    def _build_prompt(self, state) -> str:
        observations = []

        for step in state.steps:
            observations.append(
                {
                    "step": step.step_id,
                    "action": step.action,
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

    def _parse(self, raw: str) -> ReasoningDecision:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Reasoner 返回的不是有效 JSON：{exc}\n"
                f"原始输出：{raw}"
            ) from exc

        action = str(
            data.get("action", "")
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
                f"Reasoner 返回了未知 action：{action}"
            )

        arguments = data.get("arguments")

        if not isinstance(arguments, dict):
            arguments = {}

        claims = data.get("claims")

        if not isinstance(claims, list):
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
            goal=str(data.get("goal", "")),
            task_type=str(
                data.get("task_type", "")
            ),
            domain=str(data.get("domain", "")),
            claims=claims,
            finish_reason=str(
                data.get("finish_reason", "")
            ),
        )