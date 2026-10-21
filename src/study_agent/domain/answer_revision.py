"""Answer-version and answer-coverage contracts.

These records are intentionally separate from TaskDossier: a task snapshot
describes source material and requirements, while answer revisions describe
what was actually answered and which requirements that specific version met.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from study_agent.domain.task_dossier import AnswerClaim, TaskDossier


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _unique_ids(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} contains duplicate IDs")


@dataclass(frozen=True)
class AnswerRevision:
    """An immutable answer snapshot tied to one immutable TaskDossier version."""

    user_id: str
    task_id: str
    revision_id: str
    dossier_version: int
    answer_text: str
    claims: tuple[AnswerClaim, ...] = ()
    knowledge_point_ids: tuple[str, ...] = ()
    parent_revision_id: str | None = None
    revision_number: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("user_id", "task_id", "revision_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.dossier_version, int) or isinstance(self.dossier_version, bool) or self.dossier_version < 1:
            raise ValueError("dossier_version must be a positive integer")
        if not isinstance(self.revision_number, int) or isinstance(self.revision_number, bool) or self.revision_number < 1:
            raise ValueError("revision_number must be a positive integer")
        if self.schema_version != 1:
            raise ValueError(f"unsupported AnswerRevision schema_version: {self.schema_version}")
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise ValueError("answer_text must be a non-empty string")

        parent = str(self.parent_revision_id or "").strip() or None
        object.__setattr__(self, "parent_revision_id", parent)
        if self.revision_number == 1 and parent is not None:
            raise ValueError("the first answer revision cannot have a parent")
        if self.revision_number > 1 and parent is None:
            raise ValueError("revised answers must reference their parent revision")
        if parent == self.revision_id:
            raise ValueError("an answer revision cannot be its own parent")

        claims = tuple(self.claims)
        if any(not isinstance(item, AnswerClaim) for item in claims):
            raise TypeError("claims must contain AnswerClaim records")
        _unique_ids(tuple(item.id for item in claims), "answer revision claims")
        object.__setattr__(self, "claims", claims)

        point_ids = tuple(_required_text(value, "knowledge point id") for value in self.knowledge_point_ids)
        _unique_ids(point_ids, "knowledge_point_ids")
        object.__setattr__(self, "knowledge_point_ids", point_ids)

    @property
    def content_sha256(self) -> str:
        """Hash the exact UTF-8 answer text; this is integrity metadata, not proof of truth."""
        return "sha256:" + hashlib.sha256(self.answer_text.encode("utf-8")).hexdigest()

    def validate_against(self, dossier: TaskDossier) -> None:
        if (self.user_id, self.task_id, self.dossier_version) != (
            dossier.user_id, dossier.task_id, dossier.version
        ):
            raise ValueError("answer revision scope/version does not match the TaskDossier")
        fact_ids = {item.id for item in dossier.facts}
        requirement_ids = {item.id for item in dossier.requirements}
        for claim in self.claims:
            missing_facts = sorted(set(claim.source_fact_ids) - fact_ids)
            missing_requirements = sorted(set(claim.requirement_ids) - requirement_ids)
            if missing_facts or missing_requirements:
                missing = missing_facts + missing_requirements
                raise ValueError(
                    f"answer claim {claim.id} references IDs outside the dossier: {', '.join(missing)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "task_id": self.task_id,
            "revision_id": self.revision_id,
            "dossier_version": self.dossier_version,
            "answer_text": self.answer_text,
            "claims": [
                {
                    "id": claim.id,
                    "statement": claim.statement,
                    "claim_type": claim.claim_type,
                    "source_fact_ids": list(claim.source_fact_ids),
                    "requirement_ids": list(claim.requirement_ids),
                    "assumptions": list(claim.assumptions),
                    "answer_span": claim.answer_span,
                }
                for claim in self.claims
            ],
            "knowledge_point_ids": list(self.knowledge_point_ids),
            "parent_revision_id": self.parent_revision_id,
            "revision_number": self.revision_number,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerRevision":
        if not isinstance(data, dict):
            raise TypeError("AnswerRevision data must be a dictionary")
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
            revision_id=data.get("revision_id", ""),
            dossier_version=data.get("dossier_version", 1),
            answer_text=data.get("answer_text", ""),
            claims=claims,
            knowledge_point_ids=tuple(data.get("knowledge_point_ids", [])),
            parent_revision_id=data.get("parent_revision_id"),
            revision_number=data.get("revision_number", 1),
            schema_version=data.get("schema_version", 1),
        )


class CoverageOutcome(str, Enum):
    SATISFIED = "satisfied"
    OMITTED = "omitted"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CoverageEntry:
    """The assessment of one requirement in one particular answer revision."""

    user_id: str
    task_id: str
    answer_revision_id: str
    dossier_version: int
    requirement_id: str
    outcome: CoverageOutcome
    rationale: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("user_id", "task_id", "answer_revision_id", "requirement_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.dossier_version, int) or isinstance(self.dossier_version, bool) or self.dossier_version < 1:
            raise ValueError("dossier_version must be a positive integer")
        if not isinstance(self.outcome, CoverageOutcome):
            object.__setattr__(self, "outcome", CoverageOutcome(self.outcome))
        refs = tuple(_required_text(ref, "evidence reference") for ref in self.evidence_refs)
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class CoverageLedger:
    """Coverage entries tied to one answer revision; missing entries remain unresolved."""

    user_id: str
    task_id: str
    answer_revision_id: str
    dossier_version: int
    entries: tuple[CoverageEntry, ...] = ()

    def __post_init__(self) -> None:
        for name in ("user_id", "task_id", "answer_revision_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.dossier_version, int) or isinstance(self.dossier_version, bool) or self.dossier_version < 1:
            raise ValueError("dossier_version must be a positive integer")
        entries = tuple(self.entries)
        if any(not isinstance(item, CoverageEntry) for item in entries):
            raise TypeError("entries must contain CoverageEntry records")
        requirement_ids = tuple(item.requirement_id for item in entries)
        _unique_ids(requirement_ids, "coverage ledger requirements")
        for item in entries:
            if (
                item.user_id != self.user_id
                or item.task_id != self.task_id
                or item.answer_revision_id != self.answer_revision_id
                or item.dossier_version != self.dossier_version
            ):
                raise ValueError("coverage entry scope/version does not match its ledger")
        object.__setattr__(self, "entries", entries)

    def validate_against(
        self,
        dossier: TaskDossier,
        revision: AnswerRevision,
        *,
        require_all_critical: bool = False,
    ) -> tuple[str, ...]:
        revision.validate_against(dossier)
        if (
            self.user_id != revision.user_id
            or self.task_id != revision.task_id
            or self.answer_revision_id != revision.revision_id
            or self.dossier_version != revision.dossier_version
        ):
            raise ValueError("coverage ledger does not match the answer revision")
        requirement_ids = {item.id for item in dossier.requirements}
        unknown = sorted({item.requirement_id for item in self.entries} - requirement_ids)
        if unknown:
            raise ValueError(f"coverage ledger references unknown requirements: {', '.join(unknown)}")
        covered = {item.requirement_id for item in self.entries}
        missing_critical = tuple(
            item.id for item in dossier.requirements if item.critical and item.id not in covered
        )
        if require_all_critical and missing_critical:
            raise ValueError(
                "critical requirements lack answer coverage: " + ", ".join(missing_critical)
            )
        return missing_critical
