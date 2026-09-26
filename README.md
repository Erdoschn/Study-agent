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
