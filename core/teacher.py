import json


class Teacher:
    """
    最终教学回答生成器。
    """

    SYSTEM_PROMPT = """
你是 Study Agent 的教学引擎。

你必须根据：
- 用户原问题
- Agent 的任务理解
- 学生状态
- Agent 执行过程
- 搜索证据
- Claim 验证结果

生成最终学习回答。

要求：

1. 先回答用户真正的问题。
2. 如果用户理解存在错误，要明确指出错误在哪里。
3. 区分：
   - 外部事实
   - 数学/逻辑推导
   - 实验结果
   - 模型推断
4. 外部事实尽量给出来源链接。
5. 没有证据支持的内容不要伪装成确定事实。
6. 搜索资料只能作为证据，不能因为搜到了就自动认为正确。
7. 如果证据不足，明确说明不确定。
8. 根据 Student Model 调整解释深度。
9. 不要输出模型隐藏思维链。
10. 可以给出简洁的“为什么这样判断”的决策摘要。

最终答案应以正常教学文本输出。
"""

    def __init__(self, client):
        self.client = client

    def generate(self, state) -> str:
        payload = {
            "question": state.question,
            "goal": state.goal,
            "task_type": state.task_type,
            "domain": state.domain,
            "student": {
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
            "steps": [
                {
                    "step": step.step_id,
                    "action": step.action,
                    "tool": step.tool,
                    "reasoning_summary": (
                        step.reasoning_summary
                    ),
                    "observation": step.observation,
                    "success": step.success,
                    "error": step.error,
                }
                for step in state.steps
            ],
            "evidence": state.evidence,
            "claims": state.claims,
        }

        return self.client.generate(
            self.SYSTEM_PROMPT,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
        )