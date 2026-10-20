# Architecture Migration Plan

This plan moves the existing repository toward the modular architecture without a risky big-bang rewrite.

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

- Extract dependency construction from `main.py` and scattered entry points into a single composition root.
- Introduce explicit application use cases for agent execution, assessment and learner-state queries.
- Keep CLI, HTTP, MCP and Web Model as thin adapters.
- Keep the current `StudyAgent.run` entry point as a compatibility facade while callers migrate.

**Exit criteria:** entry points depend on application contracts and do not create their own competing reasoning loops.

## Phase 3 — Ports and provider adapters

- Define narrow model, search, learner-store, event-sink and clock/cancellation contracts.
- Adapt existing model registry/router/factory and `tools/search/` implementations behind those contracts.
- Normalize transport, empty, low-relevance, timeout, cancellation and provider errors.
- Ensure retries are bounded and only applied to retryable failures.

**Exit criteria:** reasoning and domain tests can use fakes without network calls or real provider configuration.

## Phase 4 — Domain extraction

Extract stable domain rules from broad modules in small, test-protected slices:

1. Learner state and mastery/assessment rules.
2. Evidence, relevance, coverage, claims and verification.
3. Task/goal value objects and assessment contracts.
4. Application lifecycle and loop policy, preserving the simple `Decide → Act → Observe → Decide` model.

Do not split modules solely to reduce line count. Split when there is an independent reason to change, test or replace a responsibility.

**Exit criteria:** domain rules do not import HTTP, provider SDKs, UI or filesystem/environment-specific code.

## Phase 5 — Compatibility package migration

- Introduce the target `src/study_agent/` package by bounded context.
- Move one slice at a time, leaving `core.*` and `tools.search.*` compatibility re-exports where needed.
- Migrate callers and remove compatibility layers only after repository-wide import search and CI prove there are no remaining consumers.
- Keep `coder/` a separate bounded subsystem; integrate through explicit bridge/contracts.

**Exit criteria:** package imports reflect ownership, compatibility tests pass, and there is no circular dependency.

## Phase 6 — Test and research architecture

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
