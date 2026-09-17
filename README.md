## Agent Loop

```mermaid
sequenceDiagram
    actor User
    participant Agent as StudyAgent
    participant Reasoner as Reasoner
    participant Router as ModelRouter
    participant LLM as LLM
    participant Tool as ToolExecutor
    participant Obs as Observation
    participant Teacher as Teacher

    User->>Agent: 学习问题
    Agent->>Reasoner: 当前状态

    loop Reason → Act → Observe
        Reasoner->>Router: 选择模型
        Router->>LLM: 调用模型
        LLM-->>Reasoner: Action

        alt 需要工具
            Reasoner->>Tool: 执行 Action
            Tool-->>Obs: 返回结果
            Obs-->>Reasoner: Observation
        else 不需要工具
            Reasoner-->>Reasoner: 内部推理
        end

        Reasoner->>Reasoner: 重新评估
    end

    Reasoner->>Teacher: ANSWER
    Teacher->>Router: 选择回答模型
    Router->>LLM: 生成答案
    LLM-->>Teacher: 教学回答
    Teacher-->>Agent: Final Answer
    Agent-->>User: 最终回答