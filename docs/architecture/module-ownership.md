# Current Module Ownership Map

This map describes the current repository as it exists today. It is intentionally honest about broad/overlapping modules; it does not imply the target structure has already been implemented.

| Current path | Current responsibility | Target owner | Notes / migration risk |
|---|---|---|---|
| `core/agent.py` | Request lifecycle, task analysis, loop setup, teacher handoff and state coordination | `application` + `reasoning` | High-risk seam; characterize behavior before extraction |
| `core/tool_loop.py` | Tool schemas, allow-list, execution, observations, evidence and search coordination | `reasoning` + `application` + ports | Mixed policy and execution; split by responsibility, not by line count |
| `core/reasoner.py` | Model-backed decision generation and output parsing | `reasoning` | Keep parsing/validation deterministic and independently tested |
| `core/task_analyzer.py`, `core/planner.py`, `core/goal.py` | Task classification, plan and goal logic | `reasoning` / domain task concepts | Separate model-generated hypotheses from validated domain values |
| `core/state.py`, `core/knowledge_graph.py`, `core/assessment.py` | Learner state, knowledge relationships and assessment updates | `domain/learner`, `domain/assessment` | User-scoped persistence is a hard invariant |
| `core/evidence.py` | Evidence normalization, relevance, coverage and verification | `domain/evidence` + infrastructure adapters | Preserve distinction between coverage and proof |
| `core/teacher.py`, `core/teaching_validator.py`, `core/assessment_generator.py` | Teaching generation, validation and assessment generation | Application use cases + domain policies + model port | Avoid coupling teaching policy to one provider |
| `core/model_registry.py`, `core/model_router.py`, `core/model_factory.py`, `core/model_capabilities.py`, `core/web_model.py` | Model discovery, routing, client construction, capability metadata and browser-backed model adapter | `infrastructure/models` + interface adapter | Keep explicit user model selection authoritative; remote discovery is not an execution permission gate |
| `tools/search/` | Search interfaces, transport, arXiv/Wikipedia providers and routing | Ports + `infrastructure/search` | Normalize transport failure, empty result and low relevance separately |
| `core/context_management.py`, `core/user_identity.py`, `core/storage_security.py`, `core/http_security.py`, `core/redaction.py` | Identity, context scope, security and redaction | Application/security infrastructure | Cross-cutting code needs explicit contracts and tests |
| `core/__debug__.py`, `core/operation_timing.py`, `core/token_usage.py` | Debug trace, timings and usage metrics | `infrastructure/observability` | DebugTracer remains separate from the main business architecture |
| `config/` | Configuration loading and prompt/model/runtime assets | Composition root + configuration adapter | Validate at boundary; pass settings explicitly |
| `main.py` | CLI and dependency construction | `interfaces/cli` + composition root | Do not let each entry point build a different dependency graph |
| `integrations/study_agent_client.py`, `integrations/mcp_server.py`, web APIs | Client/MCP/API protocol adaptation | `interfaces` | Must call shared application use cases |
| `web/` | Static browser UI | UI | Keep UI state/formatting out of domain rules |
| `coder/` | Coding agent, sandbox, workspace and memory | Separate bounded subsystem | Integrate through the explicit Study Agent bridge |
| `test/`, `tests/` | Existing tests in two roots | Unit/integration/architecture/evaluation suites | Do not move until test discovery and CI behavior are characterized |

## Initial high-value seams

1. Extract a composition root before moving domain modules. This makes dependencies explicit and avoids constructing clients in multiple entry points.
2. Introduce normalized operation outcomes for provider/search calls. This addresses a class of silent-success and blind-retry bugs.
3. Separate domain rules from model invocation in learner assessment and evidence evaluation.
4. Add architecture tests for dependency direction and compatibility imports.
5. Consolidate test organization only after collecting a baseline of test counts, durations and CI discovery results.

## Do not do

- Do not rename `core/` wholesale in one PR.
- Do not introduce an interface for every class.
- Do not combine architectural moves with prompt changes or model-selection behavior changes.
- Do not remove old imports until compatibility tests and repository-wide searches prove they are unused.
