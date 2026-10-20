# Architecture Migration Plan

This plan moves the existing repository toward the modular architecture without a risky big-bang rewrite.

## Current status (architecture branch)

The first structural slice is implemented and its latest CI run passed on Python 3.11 and 3.13, including the separate Coder sandbox security job. It adds `src/study_agent/` domain/ports/application/infrastructure modules, a versioned Dossier repository, counterexample audit contracts/service, a security-module compatibility move, a shared composition root, and architecture/regression tests.

This is not yet the end state described in the target package map. Existing `core/agent.py`, `core/reasoner.py`, `core/teacher.py`, `core/teaching_validator.py`, `core/tool_loop.py`, model routing and learner graph code remain the active runtime. The new Dossier/audit service is currently a testable application component; it is not yet wired into every answer-generation path, and concrete Falsifier/Verifier adapters are not yet implemented.


## Baseline observations

The current repository has useful separations already: `core/agent.py`, `core/tool_loop.py`, `core/reasoner.py`, `core/model_router.py`, `core/evidence.py`, `core/state.py`, `tools/search/`, `integrations/` and `coder/`. The problem is that these boundaries are conventions rather than enforced contracts.

The root also has two test roots (`test/` and `tests/`), a broad `core/` package, and several entry points. These are migration risks, not reasons to rename everything immediately.

## Phase 0 — Architecture contract (this change)

- Document module ownership, dependency direction, lifecycle, failure categories and migration rules.
- Keep runtime behavior and import paths unchanged.
- Record explicit compatibility requirements.

**Exit criteria:** reviewers can identify the owning layer for a change and know which dependencies are forbidden.

## Phase 1 — Baseline and architecture guardrails

- Inventory imports and identify cycles, oversized modules and concrete provider construction outside the composition root.
- Add architecture tests for forbidden dependency directions and import compatibility.
- Consolidate test configuration and establish a single documented test command; do not delete or move tests until discovery and CI behavior are verified.
- Add a lightweight module ownership map.

**Exit criteria:** architecture violations fail CI; existing tests remain discoverable and passing.

## Phase 2 — Composition root and application use cases

- [x] Extract standard agent dependency construction into `src/study_agent/bootstrap.py`; keep `main.py` builder imports as compatibility exports.
- [x] Route the standard Study Agent API and browser-backed Study Agent API builders through that composition root, retaining explicit hooks for prewarming and one-model cooldown policy.
- [ ] Introduce explicit application use cases for the full agent-run, assessment and learner-state lifecycles.
- [ ] Migrate remaining integrations and examples; keep CLI, HTTP, MCP and Web Model as thin adapters.
- Keep the current `StudyAgent.run` entry point as a compatibility facade while callers migrate.

**Exit criteria:** entry points depend on application contracts and do not create their own competing reasoning loops.

## Phase 3 — Ports and provider adapters

- [x] Define initial Dossier persistence and counterexample generator/verifier ports.
- [x] Move the storage-security implementation into `study_agent.infrastructure`, retaining `core.storage_security` as a re-export facade.
- [ ] Define narrow model, search, learner-store, event-sink and clock/cancellation contracts.
- [ ] Adapt existing model registry/router/factory and `tools/search/` implementations behind those contracts.
- Normalize transport, empty, low-relevance, timeout, cancellation and provider errors.
- Ensure retries are bounded and only applied to retryable failures.

**Exit criteria:** reasoning and domain tests can use fakes without network calls or real provider configuration.

## Phase 4 — Task Dossier and counterexample-driven auditing

This is a reliability milestone, not a prompt-only tweak. See the [Epistemic Integrity design](epistemic-integrity.md). The data contracts, SQLite adapter and injectable counterexample audit orchestrator are implemented; end-to-end integration remains open.

- [x] Define immutable, versioned, user/task-scoped Dossier source/fact/requirement/question/claim contracts, validating IDs and references.
- [x] Implement role-specific Dossier projection that exposes omitted critical facts/requirements and missing claim links.
- [x] Persist immutable Dossier versions to SQLite under a composite user/task/version key; reject overwrites, version skips and cross-user reads.
- [x] Add explicit claim-level audit outcomes and a bounded application orchestrator; a model-only judgment cannot promote a candidate to `REFUTED`.
- [ ] Replace active Reasoner/Teacher/Validator lossy handoffs with role-specific Dossier projections identifying their source Dossier version and included fact/requirement IDs.
- [ ] Implement production Falsifier and independent Verifier adapters for model-driven candidate discovery, deterministic calculators/formal tools, and reliable source retrieval.
- Reuse deterministic calculators, tests and authoritative retrieval where possible; model-generated counterexamples remain candidates until validated.
- Require actionable findings with claim locations, violated assumptions, evidence/reproduction and correction.
- Add evaluation fixtures for forgotten givens, missing conditions, false counterexamples, boundary cases, source conflicts and unresolved questions.

**Exit criteria:** prompt compression in the live agent does not silently discard critical requirements; accepted counterexamples are reproducible or source-validated; "not falsified" is never represented as proof; audit feedback is actionable and user/task/version scoped. The first two contracts/storage/orchestration portions are in place, but this phase is not complete until runtime integration and adapters pass end-to-end regression tests.

## Phase 5 — Domain extraction

Extract stable domain rules from broad modules in small, test-protected slices:

1. Learner state and mastery/assessment rules.
2. Evidence, relevance, coverage, claims and verification.
3. Task/goal value objects and assessment contracts.
4. Application lifecycle and loop policy, preserving the simple `Decide → Act → Observe → Decide` model.

Do not split modules solely to reduce line count. Split when there is an independent reason to change, test or replace a responsibility.

**Exit criteria:** domain rules do not import HTTP, provider SDKs, UI or filesystem/environment-specific code.

## Phase 6 — Compatibility package migration

- Introduce the target `src/study_agent/` package by bounded context.
- Move one slice at a time, leaving `core.*` and `tools.search.*` compatibility re-exports where needed.
- Migrate callers and remove compatibility layers only after repository-wide import search and CI prove there are no remaining consumers.
- Keep `coder/` a separate bounded subsystem; integrate through explicit bridge/contracts.

**Exit criteria:** package imports reflect ownership, compatibility tests pass, and there is no circular dependency.

## Phase 7 — Test and research architecture

- Organize tests by unit, integration, architecture and evaluation intent (physical moves are optional and must not disrupt CI).
- Add deterministic fixtures for model outputs, provider errors, cancellation, retries, user isolation and evidence quality.
- Add reproducible evaluation cases for execution strategy, learner-state updates, teaching quality and tool-selection behavior.
- Track behavioral metrics separately from infrastructure reliability metrics.

**Exit criteria:** changes are evaluated both for correctness and for whether they improve the intended learning/agent behavior.

## Rules for every migration PR

- One bounded slice per PR.
- No simultaneous architectural rewrite and unrelated behavior change.
- Add characterization tests before moving behavior.
- Preserve public entry points unless a deliberate deprecation is approved.
- Include rollback notes and a file-level summary in the PR.
- Do not claim local tests were run unless they actually were; use CI as the authoritative result.
