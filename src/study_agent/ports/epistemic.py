"""Ports for dossier persistence and independent counterexample auditing."""
from __future__ import annotations

from typing import Protocol, Sequence

from study_agent.domain.audit import CandidateValidation, CounterexampleCandidate
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
    """Persist audit reports with the owning user/task scope and dossier version."""

    def save(self, report: object) -> None:
        ...
