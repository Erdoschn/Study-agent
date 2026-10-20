# Study Agent Architecture

## Status

This document defines the target architecture and rules for future changes. The repository has completed an initial, CI-validated migration slice, **not a wholesale migration of the legacy `core/` system**.

Implemented in the current architecture branch:

- A `src/study_agent/` package with initial domain, ports, application and infrastructure layers.
- Versioned Task Dossier contracts, explicit context-loss reporting, counterexample/audit outcomes and an application-level falsification orchestrator.
- SQLite persistence keyed by user, task and dossier version; storage-security implementation moved to infrastructure while `core.storage_security` remains a compatibility facade.
- One shared composition root used by the CLI and agent-enabled API builders, with explicit hooks for browser-specific setup and routing policy.
- CI architecture tests that constrain imports inside the new package and regression tests for dossier traceability, counterexample validation, persistence isolation, composition and compatibility.

Still intentionally **not** claimed complete: the legacy Reasoner/Teacher/tool loop and existing knowledge-graph model still primarily live in `core/`; `Teacher`/`TeachingValidator` are not yet routed through the new Dossier/counterexample audit service; there is not yet a production Falsifier/Verifier adapter backed by live models or deterministic math/code checkers; and the existing test roots have not been physically consolidated. Those changes must follow in smaller migration slices instead of being hidden behind a directory rename.

Existing import paths and runtime behavior remain supported while each bounded context is migrated.

## Architectural style

Study Agent is a **modular monolith** with explicit ports and adapters. It is not split into microservices. The agent loop remains dynamic:

`Decide → Act → Observe → Update state → Decide`

The architecture separates *what the system does* from *how it connects to providers, storage, interfaces, and operating-system resources*.

## Target package map

The following is the intended destination. Migration is incremental; do not move all files in one change.

```text
src/study_agent/
  domain/             # Stable concepts and rules; no network or framework imports
    learner/          # Student state, mastery, knowledge graph concepts
    evidence/         # Evidence, claims, coverage and verification rules
    assessment/       # Assessment entities and scoring rules
    tasks/             # Task/goal concepts and value objects
  application/        # Use cases and orchestration
    run_agent.py
    assess_learner.py
    build_context.py
  reasoning/          # Task analysis, planning, reasoning decisions and loop policy
    task_analyzer.py
    planner.py
    reasoner.py
    tool_loop.py
  ports/              # Interfaces required by application/domain
    model_provider.py
    search_provider.py
    learner_store.py
    event_sink.py
  infrastructure/    # Implementations of ports
    models/            # Registry, router, client factory, provider adapters
    search/             # arXiv, Wikipedia, HTTP transport and search routing
    persistence/        # SQLite and other storage adapters
    observability/      # Debug tracing, timing, token usage, redaction
    security/           # HTTP/storage security and identity adapters
  interfaces/         # Entry points and protocol adapters
    cli/
    http/
    mcp/
    web_model/
tests/
  unit/
  integration/
  architecture/
coder/                # Separate coding-agent subsystem / bounded context
web/                  # Static browser UI
config/               # Runtime configuration and prompt assets
docs/
```

The exact filenames may change during implementation, but ownership and dependency direction must remain stable. Do not create empty packages merely to match this diagram.

## Layer responsibilities

| Layer | Owns | Must not own |
|---|---|---|
| Domain | Learner state, assessment rules, evidence semantics, task concepts | HTTP, provider SDKs, UI, environment-specific paths |
| Application | Use-case orchestration, lifecycle, transaction boundaries, cancellation and fallback policy | Provider-specific request formats or UI rendering |
| Reasoning | Task analysis, decision schema, planning and loop policy | Direct persistence, hidden global configuration, interface-specific response formatting |
| Ports | Small contracts needed by inner layers | Provider implementations or business policy |
| Infrastructure | LLM clients, search providers, storage, telemetry and security adapters | Deciding the learning policy or rewriting application workflows |
| Interfaces | CLI, HTTP/API, MCP and browser-facing adapters | Duplicating the agent loop or domain rules |
| Coder subsystem | Coding-agent workflow, sandbox and workspace operations | Direct mutation of Study Agent domain internals |

### Dependency direction

Dependencies point inward:

```text
Interfaces ──▶ Application ──▶ Domain
                    │
                    ▼
                  Ports ◀── Infrastructure
```

- Domain imports only the standard library and other domain modules.
- Application may depend on domain and ports.
- Reasoning may depend on domain, application-facing contracts and ports; it must not instantiate concrete provider clients.
- Infrastructure implements ports and may depend on external libraries.
- Interfaces call application use cases. They do not implement a second copy of the reasoning loop.
- Cross-module communication uses explicit arguments, return values, typed results or documented events—not hidden mutable globals.

## Canonical request lifecycle

1. **Ingress** validates the request, resolves the authenticated user and creates a request/task context.
2. **Application** constructs the use-case context and invokes task analysis.
3. **Execution policy** chooses direct answer, verified answer, or the reasoner/tool loop.
4. **Reasoning loop** emits a typed decision; it does not execute arbitrary tools itself.
5. **Tool execution** validates the decision against a fixed capability allow-list and calls a port.
6. **Observation** converts success, empty results, low relevance, timeout, cancellation and provider failure into explicit outcomes.
7. **State update** records evidence and learner changes with user/task scope.
8. **Teaching/validation** produces and validates the response without erasing the original draft on a recoverable failure.
9. **Egress** maps the result to CLI, HTTP, MCP or UI format.
10. **Telemetry** records structured, redacted events correlated by request/task ID.

A failure at one stage must become a typed failure or a documented fallback. It must not silently masquerade as success.

## Required contracts

- **Decision contract:** normalized action, optional tool, validated arguments, reasoning summary and decision metadata. Model output is untrusted input.
- **Tool contract:** declared name/schema, bounded execution, cancellation support where possible, and a normalized observation with explicit status.
- **Provider contract:** timeout, retryability, rate-limit handling, cancellation behavior and safe error metadata.
- **State contract:** user-scoped persistence; no learner graph or long-term memory may be shared across users by default.
- **Evidence contract:** source/provenance, relevance and coverage are separate attributes; coverage is not proof of a claim.
- **Configuration contract:** load and validate once at the boundary; pass typed settings inward. No secrets in logs or committed runtime configuration.
- **Observability contract:** stable event names and correlation IDs; diagnostics must be useful without logging private prompts, file contents, credentials or unnecessary personal data.

## Failure semantics

Every external operation should resolve to one of these categories where applicable:

- `success`
- `empty` (operation succeeded, no usable result)
- `low_relevance` (results exist but fail the relevance gate)
- `retryable_error` (timeout, transient transport error, rate limit)
- `permanent_error` (invalid configuration, unsupported operation, authorization failure)
- `cancelled`
- `deadline_exceeded`

Retries must be bounded, restricted to retryable failures, and observable. A retry must not duplicate non-idempotent state changes.

## Change rules

For every feature or bug fix:

1. Identify the owning layer and module before editing.
2. State the invariant being changed and the invariant that must remain unchanged.
3. Prefer one responsibility per change; avoid unrelated cleanup.
4. Add unit tests for local rules and integration tests for cross-layer behavior.
5. Add or update architecture tests when dependency rules change.
6. Update this document only when the architecture contract changes—not for every implementation detail.

## Epistemic integrity and answer auditing

The agent must preserve critical task details across context projection and must not equate model agreement with factual correctness. The [Epistemic Integrity design](epistemic-integrity.md) defines two first-class mechanisms:

- A versioned, user-scoped Task Dossier holding source facts, requirements, open questions and claim provenance. Prompt payloads are views of the Dossier, not replacements for source material; any truncation of critical information is explicit.
- Counterexample-driven auditing: decompose material claims, attempt to falsify them, and independently validate candidate counterexamples with evidence, deterministic computation, tests or formal tools where available.

The accepted audit outcomes distinguish `REFUTED`, `SUPPORTED_WITH_EVIDENCE`, `NOT_FALSIFIED`, `UNRESOLVED`, `NOT_TESTED` and `INVALID_COUNTEREXAMPLE`. "No counterexample found" never means "proven true". User-facing feedback must identify the exact claim, counterexample/source, violated condition and correction or unresolved check.

## Migration compatibility

Until migration is complete:

- Existing imports such as `core.reasoner` and `tools.search` remain supported.
- New code should use the new package boundary only after the corresponding module has been migrated.
- Compatibility facades may re-export moved symbols temporarily; they must not contain business logic.
- A migration PR must include import-compatibility tests and run the full CI suite.
- Do not move files and rewrite every import in a single high-risk PR.
