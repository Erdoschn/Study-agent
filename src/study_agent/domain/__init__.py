"""Framework-independent Study Agent domain types."""

from .task_dossier import (
    AnswerClaim,
    DossierFact,
    DossierView,
    FactOrigin,
    OpenQuestion,
    RequirementStatus,
    SourceReference,
    TaskDossier,
    TaskRequirement,
)
from .audit import (
    AuditOutcome,
    CandidateValidation,
    CounterexampleCandidate,
    CounterexampleMethod,
)

__all__ = [
    "AnswerClaim",
    "AuditOutcome",
    "CandidateValidation",
    "CounterexampleCandidate",
    "CounterexampleMethod",
    "DossierFact",
    "DossierView",
    "FactOrigin",
    "OpenQuestion",
    "RequirementStatus",
    "SourceReference",
    "TaskDossier",
    "TaskRequirement",
]
