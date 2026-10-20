from __future__ import annotations

import json

import pytest

from study_agent.domain.task_dossier import (
    AnswerClaim,
    DossierFact,
    FactOrigin,
    OpenQuestion,
    RequirementStatus,
    SourceReference,
    TaskDossier,
    TaskRequirement,
)


def _dossier() -> TaskDossier:
    source = SourceReference(
        id="source-1",
        kind="uploaded_document",
        locator="assignment.pdf#page=2",
        content_hash="sha256:abc",
        version="1",
    )
    facts = (
        DossierFact(
            id="fact-voltage",
            statement="The source gives a voltage of 12 V.",
            origin=FactOrigin.SOURCE_EXPLICIT,
            source_refs=("source-1",),
            critical=True,
        ),
        DossierFact(
            id="fact-assumption",
            statement="The capacitor is disconnected from its source.",
            origin=FactOrigin.USER_STATED,
            source_refs=("source-1",),
            critical=True,
        ),
    )
    requirements = (
        TaskRequirement(
            id="req-units",
            statement="Express the answer in N/C.",
            source_refs=("source-1",),
            critical=True,
        ),
    )
    claims = (
        AnswerClaim(
            id="claim-field",
            statement="The electric field magnitude is E.",
            claim_type="physics",
            source_fact_ids=("fact-voltage", "fact-assumption"),
            requirement_ids=("req-units",),
            assumptions=("Uniform field between parallel plates",),
            answer_span="E = ... N/C",
        ),
    )
    return TaskDossier(
        user_id="alice",
        task_id="task-42",
        version=1,
        sources=(source,),
        facts=facts,
        requirements=requirements,
        open_questions=(OpenQuestion("question-area", "Is the plate separation specified?"),),
        claims=claims,
    )


def test_task_dossier_round_trip_preserves_origin_requirements_and_scope() -> None:
    dossier = _dossier()
    restored = TaskDossier.from_dict(json.loads(json.dumps(dossier.to_dict())))

    assert restored == dossier
    assert restored.user_id == "alice"
    assert restored.facts[0].origin is FactOrigin.SOURCE_EXPLICIT
    assert restored.requirements[0].status is RequirementStatus.OPEN


def test_task_dossier_rejects_missing_source_references() -> None:
    with pytest.raises(ValueError, match="references missing IDs"):
        TaskDossier(
            user_id="alice",
            task_id="task-42",
            version=1,
            facts=(
                DossierFact(
                    id="fact-1",
                    statement="A supposedly sourced fact",
                    origin=FactOrigin.SOURCE_EXPLICIT,
                    source_refs=("missing-source",),
                ),
            ),
        )


def test_task_dossier_rejects_missing_claim_fact_links() -> None:
    with pytest.raises(ValueError, match="references missing IDs"):
        TaskDossier(
            user_id="alice",
            task_id="task-42",
            version=1,
            claims=(
                AnswerClaim(
                    id="claim-1",
                    statement="A claim",
                    claim_type="factual",
                    source_fact_ids=("missing-fact",),
                ),
            ),
        )


def test_projection_surfaces_omitted_critical_facts_and_requirements() -> None:
    dossier = _dossier()
    view = dossier.project(fact_ids={"fact-voltage"}, requirement_ids=set())

    assert view.has_critical_context_loss is True
    assert view.missing_critical_fact_ids == ("fact-assumption",)
    assert view.missing_critical_requirement_ids == ("req-units",)
    assert view.missing_claim_fact_links == ("claim-field:fact-assumption",)
    assert view.missing_claim_requirement_links == ("claim-field:req-units",)


def test_dossier_versions_must_be_positive_and_user_scoped() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TaskDossier(user_id="alice", task_id="task", version=0)

    with pytest.raises(ValueError, match="user_id"):
        TaskDossier(user_id="", task_id="task", version=1)



def test_source_explicit_fact_requires_source_provenance():
    with pytest.raises(ValueError, match="source reference"):
        DossierFact(
            id="fact-unlinked",
            statement="A fact said to be explicitly in the source",
            origin=FactOrigin.SOURCE_EXPLICIT,
        )
