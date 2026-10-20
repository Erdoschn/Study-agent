from __future__ import annotations

import pytest

from study_agent.application.counterexample_audit import (
    CounterexampleAuditService,
    FalsificationPolicy,
)
from study_agent.domain.audit import (
    AuditOutcome,
    CandidateValidation,
    CounterexampleCandidate,
    CounterexampleMethod,
)
from study_agent.domain.task_dossier import AnswerClaim, DossierFact, FactOrigin, TaskDossier


def _fixture():
    claim = AnswerClaim(
        id="claim-all",
        statement="All positive integers are even.",
        claim_type="mathematical",
        source_fact_ids=("fact-domain",),
    )
    dossier = TaskDossier(
        user_id="user-a",
        task_id="task-a",
        version=3,
        facts=(
            DossierFact(
                id="fact-domain",
                statement="The claim quantifies over positive integers.",
                origin=FactOrigin.USER_STATED,
            ),
        ),
        claims=(claim,),
    )
    candidate = CounterexampleCandidate(
        id="candidate-1",
        target_claim_id=claim.id,
        scenario="Use n = 1.",
        premises_satisfied=("1 is a positive integer",),
        predicted_result="1 is odd",
        contradiction="The claimed universal statement requires 1 to be even.",
        reproduction={"input": {"n": 1}, "observed": "odd"},
    )
    return dossier, claim, candidate


class Generator:
    def __init__(self, candidates):
        self.candidates = candidates

    def find_counterexamples(self, dossier, claim):
        return self.candidates


class Verifier:
    def __init__(self, validation):
        self.validation = validation
        self.calls = 0

    def validate(self, dossier, claim, candidate):
        self.calls += 1
        return self.validation


def test_a_model_judgment_alone_cannot_refute_a_claim() -> None:
    dossier, claim, candidate = _fixture()
    with pytest.raises(ValueError, match="non-model-only"):
        CandidateValidation(
            candidate_id=candidate.id,
            outcome=AuditOutcome.REFUTED,
            method=CounterexampleMethod.MODEL_JUDGMENT,
            premises_hold=True,
            result_reproduced=True,
            contradiction_confirmed=True,
            rationale="The model agrees that this is a counterexample.",
        )


def test_independently_reproduced_counterexample_refutes_claim() -> None:
    dossier, claim, candidate = _fixture()
    validation = CandidateValidation(
        candidate_id=candidate.id,
        outcome=AuditOutcome.REFUTED,
        method=CounterexampleMethod.CALCULATION,
        premises_hold=True,
        result_reproduced=True,
        contradiction_confirmed=True,
        reproduction={"input": {"n": 1}, "observed": "odd", "checker": "integer-parity"},
        rationale="n=1 satisfies the premise and falsifies the universal claim.",
    )
    service = CounterexampleAuditService(Generator([candidate]), Verifier(validation))
    result = service.audit_claim(dossier, claim)

    assert result.outcome is AuditOutcome.REFUTED
    assert "n = 1" in result.finding
    assert result.is_proven_true is False


def test_no_counterexample_is_not_reported_as_proof() -> None:
    dossier, claim, _candidate = _fixture()
    result = CounterexampleAuditService(Generator([]), Verifier(None)).audit_claim(dossier, claim)

    assert result.outcome is AuditOutcome.NOT_FALSIFIED
    assert result.is_proven_true is False
    assert "not proof" in result.finding


def test_invalid_candidate_does_not_become_a_confirmed_error() -> None:
    dossier, claim, candidate = _fixture()
    validation = CandidateValidation(
        candidate_id=candidate.id,
        outcome=AuditOutcome.INVALID_COUNTEREXAMPLE,
        method=CounterexampleMethod.TEST,
        premises_hold=False,
        result_reproduced=True,
        contradiction_confirmed=False,
        reproduction={"input": {"n": -1}, "observed": "odd"},
        rationale="Candidate does not satisfy the positive-integer premise.",
    )
    result = CounterexampleAuditService(
        Generator([candidate]),
        Verifier(validation),
    ).audit_claim(dossier, claim)

    assert result.outcome is AuditOutcome.NOT_FALSIFIED
    assert result.validations[0].outcome is AuditOutcome.INVALID_COUNTEREXAMPLE


def test_verifier_failure_is_recorded_as_unresolved_instead_of_crashing_audit() -> None:
    dossier, claim, candidate = _fixture()

    class FailingVerifier:
        def validate(self, dossier, claim, candidate):
            raise TimeoutError("verification backend timeout")

    result = CounterexampleAuditService(
        Generator([candidate]), FailingVerifier()
    ).audit_claim(dossier, claim)

    assert result.outcome is AuditOutcome.UNRESOLVED
    assert result.validations[0].outcome is AuditOutcome.UNRESOLVED
    assert "TimeoutError" in result.validations[0].rationale


def test_cross_claim_counterexample_is_rejected_without_verifier_call() -> None:
    dossier, claim, candidate = _fixture()
    wrong_target = CounterexampleCandidate(
        id="candidate-wrong",
        target_claim_id="another-claim",
        scenario=candidate.scenario,
        premises_satisfied=candidate.premises_satisfied,
        predicted_result=candidate.predicted_result,
        contradiction=candidate.contradiction,
    )
    verifier = Verifier(None)
    result = CounterexampleAuditService(
        Generator([wrong_target]), verifier
    ).audit_claim(dossier, claim)

    assert result.outcome is AuditOutcome.NOT_FALSIFIED
    assert result.validations[0].outcome is AuditOutcome.INVALID_COUNTEREXAMPLE
    assert verifier.calls == 0


def test_audit_is_bounded_and_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        FalsificationPolicy(max_candidates_per_claim=0)
