"""Versioned, source-traceable task facts and requirement ledger.

This module intentionally has no model, web, database, or UI dependencies.
The dossier stores normalized facts and references; it does not replace the
original source material or promote model-generated guesses to source facts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class FactOrigin(str, Enum):
    SOURCE_EXPLICIT = "source_explicit"
    USER_STATED = "user_stated"
    DERIVED = "derived"
    MODEL_INFERENCE = "model_inference"


class RequirementStatus(str, Enum):
    OPEN = "open"
    SATISFIED = "satisfied"
    UNRESOLVED = "unresolved"
    CONFLICTED = "conflicted"
    OMITTED = "omitted"
    NOT_APPLICABLE = "not_applicable"


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _unique_ids(items: tuple[Any, ...], field_name: str) -> None:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field_name} contains duplicate IDs")


@dataclass(frozen=True)
class SourceReference:
    """Pointer to original user/source material; never a model-created citation."""

    id: str
    kind: str
    locator: str
    content_hash: str = ""
    version: str = ""

    def __post_init__(self) -> None:
        _required_text(self.id, "source reference id")
        _required_text(self.kind, "source reference kind")
        _required_text(self.locator, "source reference locator")


@dataclass(frozen=True)
class DossierFact:
    """A normalized fact with provenance and an explicit origin classification."""

    id: str
    statement: str
    origin: FactOrigin
    source_refs: tuple[str, ...] = ()
    critical: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        _required_text(self.id, "fact id")
        _required_text(self.statement, "fact statement")
        if not isinstance(self.origin, FactOrigin):
            object.__setattr__(self, "origin", FactOrigin(self.origin))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        if any(not str(ref).strip() for ref in self.source_refs):
            raise ValueError("fact source references must be non-empty IDs")


@dataclass(frozen=True)
class TaskRequirement:
    """Explicit user requirement or source-imposed constraint."""

    id: str
    statement: str
    source_refs: tuple[str, ...] = ()
    critical: bool = True
    status: RequirementStatus = RequirementStatus.OPEN
    notes: str = ""

    def __post_init__(self) -> None:
        _required_text(self.id, "requirement id")
        _required_text(self.statement, "requirement statement")
        if not isinstance(self.status, RequirementStatus):
            object.__setattr__(self, "status", RequirementStatus(self.status))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        if any(not str(ref).strip() for ref in self.source_refs):
            raise ValueError("requirement source references must be non-empty IDs")


@dataclass(frozen=True)
class OpenQuestion:
    """A missing, ambiguous, conflicting, or unreadable task detail."""

    id: str
    question: str
    source_refs: tuple[str, ...] = ()
    critical: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        _required_text(self.id, "open question id")
        _required_text(self.question, "open question")
        object.__setattr__(self, "source_refs", tuple(self.source_refs))


@dataclass(frozen=True)
class AnswerClaim:
    """Atomic candidate-answer claim connected to task facts and requirements."""

    id: str
    statement: str
    claim_type: str
    source_fact_ids: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    answer_span: str = ""

    def __post_init__(self) -> None:
        _required_text(self.id, "claim id")
        _required_text(self.statement, "claim statement")
        _required_text(self.claim_type, "claim type")
        object.__setattr__(self, "source_fact_ids", tuple(self.source_fact_ids))
        object.__setattr__(self, "requirement_ids", tuple(self.requirement_ids))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True)
class TaskDossier:
    """Immutable snapshot of a task's source-linked facts and requirements."""

    user_id: str
    task_id: str
    version: int
    sources: tuple[SourceReference, ...] = ()
    facts: tuple[DossierFact, ...] = ()
    requirements: tuple[TaskRequirement, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()
    claims: tuple[AnswerClaim, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _required_text(self.user_id, "user_id")
        _required_text(self.task_id, "task_id")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("dossier version must be a positive integer")
        if self.schema_version != 1:
            raise ValueError(f"unsupported TaskDossier schema_version: {self.schema_version}")

        for field_name in ("sources", "facts", "requirements", "open_questions", "claims"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
            _unique_ids(getattr(self, field_name), field_name)

        source_ids = {item.id for item in self.sources}
        fact_ids = {item.id for item in self.facts}
        requirement_ids = {item.id for item in self.requirements}
        for fact in self.facts:
            self._validate_refs(fact.source_refs, source_ids, f"fact {fact.id}")
        for requirement in self.requirements:
            self._validate_refs(requirement.source_refs, source_ids, f"requirement {requirement.id}")
        for question in self.open_questions:
            self._validate_refs(question.source_refs, source_ids, f"question {question.id}")
        for claim in self.claims:
            self._validate_refs(claim.source_fact_ids, fact_ids, f"claim {claim.id} facts")
            self._validate_refs(claim.requirement_ids, requirement_ids, f"claim {claim.id} requirements")

    @staticmethod
    def _validate_refs(refs: tuple[str, ...], available: set[str], owner: str) -> None:
        missing = sorted(set(refs) - available)
        if missing:
            raise ValueError(f"{owner} references missing IDs: {', '.join(missing)}")

    def project(
        self,
        *,
        fact_ids: set[str] | None = None,
        requirement_ids: set[str] | None = None,
        claim_ids: set[str] | None = None,
    ) -> "DossierView":
        """Build a role-specific view and report critical omissions explicitly."""
        selected_facts = (
            tuple(item for item in self.facts if item.id in fact_ids)
            if fact_ids is not None else self.facts
        )
        selected_requirements = (
            tuple(item for item in self.requirements if item.id in requirement_ids)
            if requirement_ids is not None else self.requirements
        )
        selected_claims = (
            tuple(item for item in self.claims if item.id in claim_ids)
            if claim_ids is not None else self.claims
        )
        included_facts = {item.id for item in selected_facts}
        included_requirements = {item.id for item in selected_requirements}
        missing_critical_facts = tuple(
            item.id for item in self.facts if item.critical and item.id not in included_facts
        )
        missing_critical_requirements = tuple(
            item.id for item in self.requirements
            if item.critical and item.id not in included_requirements
        )
        missing_claim_fact_links = tuple(
            f"{claim.id}:{fact_id}"
            for claim in selected_claims
            for fact_id in claim.source_fact_ids
            if fact_id not in included_facts
        )
        missing_claim_requirement_links = tuple(
            f"{claim.id}:{requirement_id}"
            for claim in selected_claims
            for requirement_id in claim.requirement_ids
            if requirement_id not in included_requirements
        )
        return DossierView(
            user_id=self.user_id,
            task_id=self.task_id,
            dossier_version=self.version,
            sources=self.sources,
            facts=selected_facts,
            requirements=selected_requirements,
            open_questions=self.open_questions,
            claims=selected_claims,
            missing_critical_fact_ids=missing_critical_facts,
            missing_critical_requirement_ids=missing_critical_requirements,
            missing_claim_fact_links=missing_claim_fact_links,
            missing_claim_requirement_links=missing_claim_requirement_links,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with enum values normalized."""
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {
                    name: convert(getattr(value, name))
                    for name in value.__dataclass_fields__
                }
            return value

        return convert(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskDossier":
        if not isinstance(data, dict):
            raise TypeError("TaskDossier data must be a dictionary")
        sources = tuple(SourceReference(**item) for item in data.get("sources", []))
        facts = tuple(
            DossierFact(
                **{
                    **item,
                    "origin": FactOrigin(item["origin"]),
                    "source_refs": tuple(item.get("source_refs", [])),
                }
            )
            for item in data.get("facts", [])
        )
        requirements = tuple(
            TaskRequirement(
                **{
                    **item,
                    "status": RequirementStatus(item.get("status", RequirementStatus.OPEN.value)),
                    "source_refs": tuple(item.get("source_refs", [])),
                }
            )
            for item in data.get("requirements", [])
        )
        questions = tuple(
            OpenQuestion(**{**item, "source_refs": tuple(item.get("source_refs", []))})
            for item in data.get("open_questions", [])
        )
        claims = tuple(
            AnswerClaim(
                **{
                    **item,
                    "source_fact_ids": tuple(item.get("source_fact_ids", [])),
                    "requirement_ids": tuple(item.get("requirement_ids", [])),
                    "assumptions": tuple(item.get("assumptions", [])),
                }
            )
            for item in data.get("claims", [])
        )
        return cls(
            user_id=data.get("user_id", ""),
            task_id=data.get("task_id", ""),
            version=data.get("version", 1),
            sources=sources,
            facts=facts,
            requirements=requirements,
            open_questions=questions,
            claims=claims,
            schema_version=data.get("schema_version", 1),
        )


@dataclass(frozen=True)
class DossierView:
    """Role-specific Dossier projection with machine-readable context-loss signals."""

    user_id: str
    task_id: str
    dossier_version: int
    sources: tuple[SourceReference, ...] = ()
    facts: tuple[DossierFact, ...] = ()
    requirements: tuple[TaskRequirement, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()
    claims: tuple[AnswerClaim, ...] = ()
    missing_critical_fact_ids: tuple[str, ...] = ()
    missing_critical_requirement_ids: tuple[str, ...] = ()
    missing_claim_fact_links: tuple[str, ...] = ()
    missing_claim_requirement_links: tuple[str, ...] = ()

    @property
    def has_critical_context_loss(self) -> bool:
        return bool(
            self.missing_critical_fact_ids
            or self.missing_critical_requirement_ids
            or self.missing_claim_fact_links
            or self.missing_claim_requirement_links
        )

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {
                    name: convert(getattr(value, name))
                    for name in value.__dataclass_fields__
                }
            return value
        return convert(self)
