# Current Module Ownership Map

This map describes the current repository as it exists today. It is intentionally honest about broad/overlapping modules; it does not imply the target structure has already been implemented.

| Current path | Current responsibility | Target owner | Notes / migration risk |
|---|---|---|---|
| `core/agent.py` | Request lifecycle, task analysis, loop setup, teacher handoff and state coordination | `application` + `reasoning` | High-risk seam; characterize behavior before extraction |
| `core/tool_loop.py` | Tool schemas, allow-list, execution, observations, evidence and search coordination | `reasoning` + `application` + ports | Mixed policy and execution; split by responsibility, not by line count |
| `core/reasoner.py` | Model-backed decision generation and output parsing | `reasoning` | Keep parsing/validation deterministic and independently tested |
| `core/task_analyzer.py`, `core/planner.py`, `core/goal.py` | Task classification, plan and goal logic | `reasoning` / domain task concepts | Separate model-generated hypotheses from validated domain values |
| `core/state.py`, `core/knowledge_graph.py`, `core/assessment.py` | Active runtime learner state, knowledge relationships and assessment updates | `domain/learner`, `domain/assessment` | Still legacy runtime; user-scoped persistence is a hard invariant |
| `core/evidence.py` | Evidence normalization, relevance, coverage and verification | `domain/evidence` + infrastructure adapters | Preserve distinction between coverage and proof; a Dossier must hold source-linked facts and requirements independently of prompt summaries. |
| `core/teacher.py`, `core/teaching_validator.py`, `core/assessment_generator.py` | Teaching generation, validation and assessment generation | Application use cases + domain policies + model port | Existing validator returns model-judged PASS/REVISE/UNCERTAIN; it does not yet perform validated counterexample-driven auditing. Preserve answer/source traceability; never treat model PASS as proof. |
| `core/model_registry.py`, `core/model_router.py`, `core/model_factory.py`, `core/model_capabilities.py`, `core/web_model.py` | Model discovery, routing, client construction, capability metadata and browser-backed model adapter | `infrastructure/models` + interface adapter | Keep explicit user model selection authoritative; remote discovery is not an execution permission gate |
| `tools/search/` | Search interfaces, transport, arXiv/Wikipedia providers and routing | Ports + `infrastructure/search` | Normalize transport failure, empty result and low relevance separately |
| `core/context_management.py`, `core/user_identity.py`, `core/http_security.py`, `core/redaction.py` | Active identity, context scope, HTTP security and redaction | Application/security infrastructure | Cross-cutting code needs explicit contracts and tests |\n| `src/study_agent/infrastructure/storage_security.py` / `core/storage_security.py` | SQLite path and private-directory security implementation / legacy re-export facade | `infrastructure/security` | Moved implementation is protected by compatibility and existing storage-security regression tests |
| `core/__debug__.py`, `core/operation_timing.py`, `core/token_usage.py` | Debug trace, timings and usage metrics | `infrastructure/observability` | DebugTracer remains separate from the main business architecture |
| `config/` | Configuration loading and prompt/model/runtime assets | Composition root + configuration adapter | Validate at boundary; pass settings explicitly |
| `main.py` | CLI and dependency construction | `interfaces/cli` + composition root | Do not let each entry point build a different dependency graph |
| `integrations/study_agent_client.py`, `integrations/mcp_server.py`, web APIs | Client/MCP/API protocol adaptation | `interfaces` | Must call shared application use cases |
| `web/` | Static browser UI | UI | Keep UI state/formatting out of domain rules |
| `coder/` | Coding agent, sandbox, workspace and memory | Separate bounded subsystem | Integrate through the explicit Study Agent bridge |
| `test/`, `tests/` | Existing tests in two roots | Unit/integration/architecture/evaluation suites | Do not move until test discovery and CI behavior are characterized |

## Implemented first-slice modules

| New path | Responsibility | Boundary/invariant |
|---|---|---|
| `src/study_agent/domain/task_dossier.py` | Immutable versioned source/fact/requirement/question/claim records and role-specific projections | References must resolve within one dossier snapshot; project() reports critical omissions instead of silently implying full context |
| `src/study_agent/domain/audit.py` | Counterexample candidate, independent validation, claim-audit and report outcomes | Model judgment alone cannot establish `REFUTED`; no outcome is represented as absolute proof |
| `src/study_agent/ports/epistemic.py` | Dossier repository, counterexample generator/verifier and audit-report contracts | Generation and validation are separate roles; persistence contract is user/task scoped |
| `src/study_agent/application/counterexample_audit.py` | Bounded orchestration of candidate generation and independent validation | Validation exceptions become `UNRESOLVED`; no generated candidate is automatically accepted |
| `src/study_agent/infrastructure/dossier_sqlite.py` | Append-only per-user/task Dossier versions | Composite key, version progression, scoped reads and hardened SQLite path |
| `src/study_agent/bootstrap.py` | Shared legacy-compatible composition root | One registry/router/factory graph, narrowly scoped runtime hooks, explicit storage path |

These modules form the new architectural boundary, but the active Teacher/Validator and full agent loop remain under `core/` until later phases migrate them.
 
## Newly identified reliability seam

The Task Dossier and counterexample-audit lifecycle are cross-cutting contracts, not extra responsibilities to place inside `Teacher`. Target ownership:

- Dossier/source facts/requirements/claim links: domain entities plus persistence ports/adapters.
- Dossier construction, projection and version invalidation: application use case.
- Falsifier role and claim decomposition: reasoning.
- Counterexample validation: application policy calling independent model/evidence/calculation/test ports; no candidate can validate itself.
- Audit findings, source spans, test outputs and calibration metrics: structured audit artifacts and observability.

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
