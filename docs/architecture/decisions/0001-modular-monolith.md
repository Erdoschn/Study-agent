# ADR 0001: Adopt a modular monolith with explicit ports and adapters

- **Status:** Accepted
- **Date:** 2026-10-20
- **Decision owners:** Study Agent maintainers

## Context

Study Agent combines agent orchestration, dynamic reasoning, model/provider routing, search, evidence handling, learner modeling, teaching, API/UI adapters, and a separate coding-agent subsystem. These capabilities need to evolve independently, but a distributed-service architecture would add operational complexity without solving the current boundary problems.

Repeated bug-fix work also shows that local tests alone do not guarantee safe dependencies or predictable failure handling.

## Decision

Keep one deployable application, but enforce logical modules and dependency direction:

- Domain rules are independent of network frameworks and provider SDKs.
- Application use cases coordinate domain rules and ports.
- Reasoning chooses actions but does not execute arbitrary operations.
- Infrastructure implements model, search, storage and observability ports.
- CLI, HTTP, MCP and browser adapters remain thin.
- Coder remains a separate bounded subsystem connected through an explicit bridge.
- Compatibility imports are maintained during incremental migration.

The canonical agent loop remains `Decide → Act → Observe → Update state → Decide`. Debug tracing remains a cross-cutting observability concern, not a node in the business architecture.

## Consequences

### Benefits

- A provider or UI change should not require changing domain rules.
- Reasoning can be tested with deterministic fake ports.
- Failures, retries, cancellation and user isolation have explicit ownership.
- New features have a clear home and dependency contract.
- Architecture violations can be checked in CI.

### Costs

- Ports and adapters add some interfaces and wiring.
- Migration requires compatibility facades and incremental import changes.
- Boundaries need enforcement; documentation alone is insufficient.

## Alternatives considered

### Continue the current structure

Rejected because module ownership and dependency direction remain implicit, making every new feature a local judgement call.

### Rewrite all packages at once

Rejected because broad import changes can break runtime entry points, tests, plugins and deployment scripts simultaneously.

### Split into microservices

Rejected because it adds network, deployment and consistency failure modes before there is evidence that separate deployment is necessary.

## Guardrails

The decision is not permission to create abstractions for every class. Introduce a port only when it isolates an external dependency, enables a meaningful test seam, or protects a stable domain boundary. Module extraction must be backed by tests and completed one bounded slice at a time.
