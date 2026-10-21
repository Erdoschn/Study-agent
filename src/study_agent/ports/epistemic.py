"""Ports for dossier persistence and independent counterexample auditing."""
from __future__ import annotations

from typing import Protocol, Sequence

from study_agent.domain.audit import AuditReport, CandidateValidation, CounterexampleCandidate
from study_agent.domain.answer_revision import AnswerRevision, CoverageLedger
from study_agent.domain.prerequisite_graph import PrerequisiteGraph
from study_agent.domain.task_dossier import AnswerClaim, TaskDossier


class TaskDossierRepository(Protocol):
    """User/task-scoped persistence contract. Implementations must enforce ownership."""

    def load_latest(self, *, user_id: str, task_id: str) -> TaskDossier | None:
        ...

    def save(self, dossier: TaskDossier) -> None:
        """Persist a new immutable version; never overwrite another user's dossier."""
        ...


class CounterexampleGenerator(Protocol):
    """Generate candidates only; implementations cannot mark candidates validated."""

    def find_counterexamples(
        self,
        dossier: TaskDossier,
        claim: AnswerClaim,
    ) -> Sequence[CounterexampleCandidate]:
        ...


class CounterexampleVerifier(Protocol):
    """Validate whether a candidate satisfies premises and actually refutes a claim."""

    def validate(
        self,
        dossier: TaskDossier,
        claim: AnswerClaim,
        candidate: CounterexampleCandidate,
    ) -> CandidateValidation:
        ...


class AuditReportRepository(Protocol):
    """Persist reports with user/task/dossier and answer-revision scope."""

    def save(self, report: AuditReport) -> None:
        ...


class AnswerRevisionRepository(Protocol):
    """User/task-scoped append-only answer revisions."""

    def load_latest(self, *, user_id: str, task_id: str) -> AnswerRevision | None:
        ...

    def load_revision(
        self, *, user_id: str, task_id: str, revision_id: str
    ) -> AnswerRevision | None:
        ...

    def save(self, revision: AnswerRevision) -> None:
        ...


class CoverageLedgerRepository(Protocol):
    """Persist coverage results under the exact answer revision they evaluate."""

    def save(self, ledger: CoverageLedger) -> None:
        ...


class PrerequisiteGraphRepository(Protocol):
    """Versioned, shared curriculum DAG repository; learner mastery is separate."""

    def load_latest(self, *, graph_id: str) -> PrerequisiteGraph | None:
        ...

    def load_version(self, *, graph_id: str, version: int) -> PrerequisiteGraph | None:
        ...

    def save(self, graph: PrerequisiteGraph) -> None:
        """Append the next immutable graph version; reject overwrites and skipped versions."""
        ...
