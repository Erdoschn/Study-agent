## Architecture

```mermaid
flowchart LR

    USER["User"] --> SA["StudyAgent"] --> LOOP["AgentToolLoop"]

    subgraph CORE["Agent Core"]
        LOOP <--> REASONER["AgentReasoner"]
        TEACHER["Teacher"]

        subgraph STATE["State"]
            AS["AgentState"]
            STUDENT["StudentState / BDI"]
            AS --> STUDENT
        end

        subgraph MODEL["Model System"]
            direction LR
            REGISTRY["ModelRegistry"] --> ROUTER["ModelRouter"] --> CLIENT["Model Client"] --> LLM["LLM API"]
        end

        LOOP <--> AS
        SA --> TEACHER
        AS --> TEACHER
    end

    subgraph EVIDENCE["Evidence"]
        EVI["EvidenceStore / Engine"]
        COVERAGE["Coverage / Relevance"]
        CLAIM["Claims / Verification"]

        EVI --> COVERAGE
        EVI --> CLAIM
    end

    subgraph TOOLS["Tool System"]
        EXECUTOR["ToolExecutor"]

        subgraph SEARCH["Search"]
            SEARCH_ROUTER["SearchRouter"] --> WIKI["Wikipedia"]
            SEARCH_ROUTER --> ARXIV["arXiv"]
        end

        CALC["Calculator"]
        VERIFY["Verification"]

        EXECUTOR --> SEARCH_ROUTER
        EXECUTOR --> CALC
        EXECUTOR --> VERIFY
    end

    CONFIG["providers.json"] --> MODEL

    REASONER <--> MODEL
    TEACHER <--> MODEL
    LOOP --> EXECUTOR
    EXECUTOR --> EVI
    EVI --> LOOP
    CLAIM --> AS
```

## Agent Reasoning Loop

```mermaid
flowchart LR

    D["Decide"]
    A["Act"]
    O["Observe"]
    FINISH["ANSWER / STOP"]

    D --> A
    A --> O
    O --> D
    O --> FINISH

    MR["Model Router"]
    TOOLS["Tool Executor"]

    MR -.-> D
    A --> TOOLS
    TOOLS --> O
```

## Debug Structure

```mermaid
flowchart TB

    TRACE["DebugTracer"]
    DEBUG_CONFIG["providers.json<br/>debug: true / false"]
    SCOPE["Debug Scope"]
    LOG["Structured Debug Logs"]

    DEBUG_CONFIG --> TRACE
    TRACE --> SCOPE
    SCOPE --> LOG

    SCOPE -.-> SA["StudyAgent"]
    SCOPE -.-> REASONER["AgentReasoner"]
    SCOPE -.-> ROUTER["ModelRouter"]
    SCOPE -.-> FACTORY["ModelClientFactory"]
    SCOPE -.-> CLIENT["OpenAICompatibleClient"]
    SCOPE -.-> EXECUTOR["ToolExecutor"]
    SCOPE -.-> SEARCH["SearchRouter"]
    SCOPE -.-> PROVIDERS["Search Providers"]
```


## Run

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The root entry point delegates to the Study Agent CLI:

```bash
python main.py
```

Copy `config/providers.example.json` to `config/providers.json` and fill in the required provider credentials.

Set `"debug": true` in `providers.json` to enable execution tracing. Debug output records model routing, search fallback, evidence assessment, verification gates, learner-state updates, knowledge-graph changes, and Teacher context sizes.

Paid models are disabled by default. Use `python main.py --allow-paid` only when paid-model use is explicitly intended.


## Current Module Responsibilities

- **TaskAnalyzer**：一次性的语义任务理解，不决定搜索来源或搜索顺序。
- **AgentReasoner**：每轮唯一的动态决策中心，负责决定 SEARCH / CALCULATE / VERIFY / ASSESS / ANSWER / STOP。
- **ToolExecutor / SearchRouter**：执行工具和搜索，不替 Reasoner 决策。
- **EvidenceEngine / EvidenceStore**：标准化、去重、相关性、coverage 与结构化 VERIFY；VERIFY 不是事实证明。
- **KnowledgeGraph**：保存概念关系与学习者状态；搜索 provenance 与 learner concept 分离。
- **AssessmentEvaluator**：对 ASSESS 的答案做保守、可重复的评分；显式 assessment 才能更新学习图谱。
- **Teacher**：在 Reasoner 最终答案之后做教学表达，并优先使用当前学生状态、知识图谱和已验证 claims。
- **ModelRegistry / ModelRouter**：维护模型运行统计、连续失败冷却和能力路由；默认禁止付费模型。
- **DebugTracer**：提供跨模块结构化执行轨迹。开启 `providers.json` 的 `debug` 后，可观察分析、决策、工具、搜索恢复、证据核查、学习状态、模型冷却和 Teacher 上下文。
