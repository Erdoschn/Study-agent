## Architecture

```mermaid
flowchart TB

    USER["User"]

    subgraph CONFIG["Configuration"]
        CFG["providers.json"]
        LOADER["config/loader.py"]
        CFG --> LOADER
    end

    subgraph AGENT["Study Agent"]
        SA["StudyAgent"]

        subgraph STATE["Agent State"]
            AS["AgentState"]
            STEP["AgentStep"]
            STUDENT["StudentState"]
        end

        LOOP["AgentToolLoop"]
        REASONER["AgentReasoner"]
        TEACHER["Teacher"]

        SA --> AS
        SA --> LOOP
        LOOP --> REASONER
        LOOP --> TEACHER
        AS --> LOOP
        LOOP --> AS
        AS --> STUDENT
    end

    subgraph MODEL["Model System"]
        REGISTRY["ModelRegistry"]
        ROUTER["ModelRouter"]
        FACTORY["ModelClientFactory"]
        CLIENT["OpenAICompatibleClient"]
        LLM["LLM API"]

        REGISTRY --> ROUTER
        ROUTER --> FACTORY
        FACTORY --> CLIENT
        CLIENT --> LLM
        LLM --> CLIENT
    end

    subgraph TOOLS["Tool System"]
        EXECUTOR["ToolExecutor"]

        subgraph SEARCH["Search"]
            SEARCH_ROUTER["SearchRouter"]
            WIKI["Wikipedia Provider"]
            ARXIV["arXiv Provider"]

            SEARCH_ROUTER --> WIKI
            SEARCH_ROUTER --> ARXIV
        end

        CALC["Calculator"]
        VERIFY["Verification Tool"]

        EXECUTOR --> SEARCH_ROUTER
        EXECUTOR --> CALC
        EXECUTOR --> VERIFY
    end

    subgraph EVIDENCE["Evidence"]
        EVI["Evidence"]
        CLAIM["Claims"]
        CHECK["Claim Verification"]
    end

    subgraph DEBUG["Debug"]
        TRACE["DebugTracer"]
    end

    USER --> SA

    LOADER --> REGISTRY
    LOADER --> FACTORY

    REASONER --> ROUTER
    TEACHER --> ROUTER

    REASONER --> EXECUTOR
    EXECUTOR --> REASONER

    SEARCH_ROUTER --> EVI
    EVI --> AS

    REASONER --> CLAIM
    CLAIM --> CHECK
    CHECK --> AS

    TEACHER --> AS
    TEACHER --> STUDENT

    TRACE -.-> SA
    TRACE -.-> REASONER
    TRACE -.-> ROUTER
    TRACE -.-> FACTORY
    TRACE -.-> CLIENT
    TRACE -.-> EXECUTOR
    TRACE -.-> SEARCH_ROUTER
    TRACE -.-> WIKI
    TRACE -.-> ARXIV
    TRACE -.-> TEACHER

    AS --> REASONER
```

## Agent Reasoning Loop

```mermaid
flowchart LR

    R["Reason"]
    A["Act"]
    O["Observe"]
    RR["Re-Reason"]

    R --> A
    A --> O
    O --> RR
    RR --> R

    MR["Model Router"]
    TOOLS["Tools"]
    FINISH["Finish"]
    TEACH["Teacher"]

    MR -.-> R
    A --> TOOLS
    TOOLS --> O

    O --> FINISH
    FINISH --> TEACH
```