"""Epistemic outcomes and counterexample validation records.

A generated counterexample remains untrusted until an independently checkable
method establishes that its premises hold and that it contradicts the claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuditOutcome(str, Enum):
    REFUTED = "REFUTED"
    SUPPORTED_WITH_EVIDENCE = "SUPPORTED_WITH_EVIDENCE"
    NOT_FALSIFIED = "NOT_FALSIFIED"
    UNRESOLVED = "UNRESOLVED"
    NOT_TESTED = "NOT_TESTED"
    INVALID_COUNTEREXAMPLE = "INVALID_COUNTEREXAMPLE"


class CounterexampleMethod(str, Enum):
    SYMBOLIC_CHECK = "symbolic_check"
    CALCULATION = "calculation"
    TEST = "test"
    FORMAL_CHECK = "formal_check"
    PRIMARY_SOURCE = "primary_source"
    CROSS_SOURCE = "cross_source"
    HUMAN_REVIEW = "human_review"
    MODEL_JUDGMENT = "model_judgment"
    NONE = "none"


@dataclass(frozen=True)
class CounterexampleCandidate:
    id: str
    target_claim_id: str
    scenario: str
    premises_satisfied: tuple[str, ...]
    predicted_result: str
    contradiction: str
    evidence_refs: tuple[str, ...] = ()
    reproduction: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "target_claim_id", "scenario", "predicted_result", "contradiction"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"counterexample {name} must not be empty")
        object.__setattr__(self, "premises_satisfied", tuple(self.premises_satisfied))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if not isinstance(self.reproduction, dict):
            raise TypeError("counterexample reproduction must be a dictionary")


@dataclass(frozen=True)
class CandidateValidation:
    candidate_id: str
    outcome: AuditOutcome
    method: CounterexampleMethod
    premises_hold: bool | None
    result_reproduced: bool | None
    contradiction_confirmed: bool | None
    evidence_refs: tuple[str, ...] = ()
    reproduction: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not str(self.candidate_id or "").strip():
            raise ValueError("candidate_id must not be empty")
        if not isinstance(self.outcome, AuditOutcome):
            object.__setattr__(self, "outcome", AuditOutcome(self.outcome))
        if not isinstance(self.method, CounterexampleMethod):
            object.__setattr__(self, "method", CounterexampleMethod(self.method))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if not isinstance(self.reproduction, dict):
            raise TypeError("validation reproduction must be a dictionary")
        if self.outcome == AuditOutcome.REFUTED and not self.confirms_refutation:
            raise ValueError(
                "REFUTED requires satisfied premises, a reproduced contradiction, "
                "and non-model-only validation evidence"
            )

    @property
    def confirms_refutation(self) -> bool:
        """True only when the method and evidence meet the verification contract."""
        if self.method in {CounterexampleMethod.MODEL_JUDGMENT, CounterexampleMethod.NONE}:
            return False
        if self.premises_hold is not True or self.contradiction_confirmed is not True:
            return False
        if self.method in {
            CounterexampleMethod.SYMBOLIC_CHECK,
            CounterexampleMethod.CALCULATION,
            CounterexampleMethod.TEST,
            CounterexampleMethod.FORMAL_CHECK,
        }:
            return self.result_reproduced is True and bool(self.reproduction)
        if self.method in {
            CounterexampleMethod.PRIMARY_SOURCE,
            CounterexampleMethod.CROSS_SOURCE,
            CounterexampleMethod.HUMAN_REVIEW,
        }:
            return self.result_reproduced is not False and bool(self.evidence_refs or self.reproduction)
        return False


@dataclass(frozen=True)
class ClaimAuditResult:
    claim_id: str
    outcome: AuditOutcome
    attempted: bool
    candidates: tuple[CounterexampleCandidate, ...] = ()
    validations: tuple[CandidateValidation, ...] = ()
    finding: str = ""
    correction: str = ""

    def __post_init__(self) -> None:
        if not str(self.claim_id or "").strip():
            raise ValueError("claim_id must not be empty")
        if not isinstance(self.outcome, AuditOutcome):
            object.__setattr__(self, "outcome", AuditOutcome(self.outcome))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "validations", tuple(self.validations))
        candidate_ids_list = [item.id for item in self.candidates]
        candidate_ids = set(candidate_ids_list)
        if len(candidate_ids_list) != len(candidate_ids):
            raise ValueError("claim audit cannot contain duplicate candidate IDs")
        validation_ids = [item.candidate_id for item in self.validations]
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError("claim audit cannot contain duplicate candidate validations")
        for candidate in self.candidates:
            if candidate.target_claim_id != self.claim_id:
                raise ValueError("candidate counterexample must target the audited claim")
        for validation in self.validations:
            if validation.candidate_id not in candidate_ids:
                raise ValueError("candidate validation references an unknown candidate")
        if self.outcome == AuditOutcome.REFUTED:
            if not any(item.confirms_refutation for item in self.validations):
                raise ValueError("REFUTED claim requires at least one independently validated counterexample")
            confirmed_ids = {item.candidate_id for item in self.validations if item.confirms_refutation}
            if not confirmed_ids.intersection(candidate_ids):
                raise ValueError("REFUTED claim validation has no matching candidate")

    @property
    def is_proven_true(self) -> bool:
        """Always false: this audit protocol does not claim absolute correctness."""
        return False


@dataclass(frozen=True)
class AuditReport:
    user_id: str
    task_id: str
    dossier_version: int
    results: tuple[ClaimAuditResult, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.user_id or "").strip() or not str(self.task_id or "").strip():
            raise ValueError("audit report must be scoped to a user and task")
        if (
            not isinstance(self.dossier_version, int)
            or isinstance(self.dossier_version, bool)
            or self.dossier_version < 1
        ):
            raise ValueError("dossier_version must be a positive integer")
        object.__setattr__(self, "results", tuple(self.results))
        ids = [result.claim_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("audit report cannot contain duplicate claim results")

    @property
    def has_confirmed_errors(self) -> bool:
        return any(result.outcome == AuditOutcome.REFUTED for result in self.results)

    @property
    def unresolved_claim_ids(self) -> tuple[str, ...]:
        return tuple(
            result.claim_id
            for result in self.results
            if result.outcome in {AuditOutcome.UNRESOLVED, AuditOutcome.NOT_TESTED}
        )
