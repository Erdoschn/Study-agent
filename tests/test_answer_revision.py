from __future__ import annotations

import pytest

from study_agent.domain.answer_revision import (
    AnswerRevision,
    CoverageEntry,
    CoverageLedger,
    CoverageOutcome,
)
from study_agent.domain.task_dossier import (
    AnswerClaim,
    DossierFact,
    FactOrigin,
    RequirementStatus,
    TaskDossier,
    TaskRequirement,
)


def _dossier() -> TaskDossier:
    return TaskDossier(
        user_id="learner-1",
        task_id="task-1",
        version=1,
        facts=(DossierFact(
            id="fact-voltage",
            statement="The voltage is 12 V.",
            origin=FactOrigin.USER_STATED,
        ),),
        requirements=(
            TaskRequirement(id="req-units", statement="Use N/C units.", critical=True),
            TaskRequirement(id="req-explain", statement="Explain the derivation.", critical=False),
        ),
    )


def _revision(dossier: TaskDossier | None = None) -> AnswerRevision:
    dossier = dossier or _dossier()
    return AnswerRevision(
        user_id=dossier.user_id,
        task_id=dossier.task_id,
        revision_id="answer-1",
        dossier_version=dossier.version,
        answer_text="The field is 12 N/C.",
        claims=(AnswerClaim(
            id="claim-field",
            statement="The field is 12 N/C.",
            claim_type="calculation",
            source_fact_ids=("fact-voltage",),
            requirement_ids=("req-units",),
        ),),
        knowledge_point_ids=("electric-field",),
    )


def test_answer_revision_hash_is_stable_and_serialization_round_trips():
    revision = _revision()
    restored = AnswerRevision.from_dict(revision.to_dict())
    assert restored == revision
    assert restored.content_sha256 == revision.content_sha256
    assert restored.content_sha256.startswith("sha256:")


def test_answer_revision_requires_explicit_parent_for_revisions():
    with pytest.raises(ValueError, match="parent revision"):
        AnswerRevision(
            user_id="learner-1",
            task_id="task-1",
            revision_id="answer-2",
            dossier_version=1,
            answer_text="revised answer",
            revision_number=2,
        )


def test_answer_revision_rejects_claim_references_outside_dossier():
    dossier = _dossier()
    revision = AnswerRevision(
        user_id="learner-1",
        task_id="task-1",
        revision_id="answer-1",
        dossier_version=1,
        answer_text="Answer",
        claims=(AnswerClaim(
            id="claim-1",
            statement="Claim",
            claim_type="factual",
            source_fact_ids=("fact-from-another-dossier",),
        ),),
    )
    with pytest.raises(ValueError, match="outside the dossier"):
        revision.validate_against(dossier)


def test_answer_revision_cannot_be_applied_to_another_user_or_dossier_version():
    revision = _revision()
    other_user = TaskDossier(
        user_id="learner-2",
        task_id="task-1",
        version=1,
        requirements=(TaskRequirement(id="req-units", statement="Use N/C units."),),
        facts=(DossierFact(id="fact-voltage", statement="12 V", origin=FactOrigin.USER_STATED),),
    )
    with pytest.raises(ValueError, match="scope/version"):
        revision.validate_against(other_user)


def test_coverage_ledger_reports_missing_critical_requirement_until_filled():
    dossier = _dossier()
    revision = _revision(dossier)
    ledger = CoverageLedger(
        user_id="learner-1",
        task_id="task-1",
        answer_revision_id="answer-1",
        dossier_version=1,
        entries=(CoverageEntry(
            user_id="learner-1",
            task_id="task-1",
            answer_revision_id="answer-1",
            dossier_version=1,
            requirement_id="req-units",
            outcome=CoverageOutcome.SATISFIED,
        ),),
    )
    assert ledger.validate_against(dossier, revision) == ()
    assert ledger.validate_against(dossier, revision, require_all_critical=True) == ()

    missing_critical = TaskDossier(
        user_id="learner-1",
        task_id="task-1",
        version=1,
        requirements=(
            TaskRequirement(id="req-units", statement="Use N/C units.", critical=True),
            TaskRequirement(id="req-significant-figures", statement="Use three significant figures.", critical=True),
        ),
    )
    matching_revision = AnswerRevision(
        user_id="learner-1",
        task_id="task-1",
        revision_id="answer-1",
        dossier_version=1,
        answer_text="12 N/C",
    )
    incomplete = CoverageLedger(
        user_id="learner-1",
        task_id="task-1",
        answer_revision_id="answer-1",
        dossier_version=1,
        entries=(CoverageEntry(
            user_id="learner-1", task_id="task-1", answer_revision_id="answer-1",
            dossier_version=1, requirement_id="req-units", outcome=CoverageOutcome.SATISFIED,
        ),),
    )
    assert incomplete.validate_against(missing_critical, matching_revision) == ("req-significant-figures",)
    with pytest.raises(ValueError, match="critical requirements"):
        incomplete.validate_against(missing_critical, matching_revision, require_all_critical=True)


def test_coverage_ledger_rejects_cross_revision_entries_and_unknown_requirement_ids():
    with pytest.raises(ValueError, match="scope/version"):
        CoverageLedger(
            user_id="learner-1", task_id="task-1", answer_revision_id="answer-1",
            dossier_version=1, entries=(CoverageEntry(
                user_id="learner-1", task_id="task-1", answer_revision_id="answer-2",
                dossier_version=1, requirement_id="req-1", outcome=CoverageOutcome.OMITTED,
            ),),
        )

    dossier = _dossier()
    revision = _revision(dossier)
    ledger = CoverageLedger(
        user_id="learner-1", task_id="task-1", answer_revision_id="answer-1",
        dossier_version=1, entries=(CoverageEntry(
            user_id="learner-1", task_id="task-1", answer_revision_id="answer-1",
            dossier_version=1, requirement_id="not-a-requirement", outcome=CoverageOutcome.SATISFIED,
        ),),
    )
    with pytest.raises(ValueError, match="unknown requirements"):
        ledger.validate_against(dossier, revision)
