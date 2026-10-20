"""Orchestrate falsification without trusting a generator to validate itself."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from study_agent.domain.audit import (
    AuditOutcome,
    AuditReport,
    CandidateValidation,
    ClaimAuditResult,
)
from study_agent.domain.task_dossier import AnswerClaim, TaskDossier
from study_agent.ports.epistemic import CounterexampleGenerator, CounterexampleVerifier


@dataclass(frozen=True)
class FalsificationPolicy:
    max_candidates_per_claim: int = 5

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_candidates_per_claim, int)
            or isinstance(self.max_candidates_per_claim, bool)
            or not 1 <= self.max_candidates_per_claim <= 20
        ):
            raise ValueError("max_candidates_per_claim must be between 1 and 20")


class CounterexampleAuditService:
    """Testable application service with independent candidate validation.

    This service intentionally does not call an LLM directly. Model/provider
    adapters implement the ports, and deterministic tools may validate candidates.
    """

    def __init__(
        self,
        generator: CounterexampleGenerator,
        verifier: CounterexampleVerifier,
        *,
        policy: FalsificationPolicy | None = None,
    ) -> None:
        self._generator = generator
        self._verifier = verifier
        self._policy = policy or FalsificationPolicy()

    def audit_claim(self, dossier: TaskDossier, claim: AnswerClaim) -> ClaimAuditResult:
        dossier_claims = {item.id: item for item in dossier.claims}
        if dossier_claims.get(claim.id) != claim:
            raise ValueError("claim must belong to the supplied dossier version")

        try:
            generated = self._generator.find_counterexamples(dossier, claim)
            # Limit iteration itself, not just the final tuple; a faulty adapter
            # must not be able to yield an unbounded sequence.
            raw_candidates = tuple(
                islice(iter(generated), self._policy.max_candidates_per_claim)
            )
        except Exception as exc:
            return ClaimAuditResult(
                claim_id=claim.id,
                outcome=AuditOutcome.UNRESOLVED,
                attempted=True,
                finding=(
                    "Counterexample generation failed: "
                    f"{type(exc).__name__}. No conclusion about correctness is implied."
                ),
            )
        candidates_list = []
        seen_candidate_ids: set[str] = set()
        malformed_candidate_count = 0
        for candidate in raw_candidates:
            if not isinstance(candidate, CounterexampleCandidate):
                malformed_candidate_count += 1
                continue
            if candidate.id in seen_candidate_ids:
                malformed_candidate_count += 1
                continue
            seen_candidate_ids.add(candidate.id)
            candidates_list.append(candidate)
        candidates = tuple(candidates_list)

        if not candidates:
            # An empty clean response is "not falsified", not proof. Malformed
            # output is a protocol failure and must remain unresolved.
            malformed = malformed_candidate_count > 0
            return ClaimAuditResult(
                claim_id=claim.id,
                outcome=AuditOutcome.UNRESOLVED if malformed else AuditOutcome.NOT_FALSIFIED,
                attempted=True,
                finding=(
                    "Counterexample generator returned malformed or duplicate candidates; audit remains unresolved."
                    if malformed else
                    "No candidate counterexample was returned; this is not proof of correctness."
                ),
            )

        validations: list[CandidateValidation] = []
        for candidate in candidates:
            if candidate.target_claim_id != claim.id:
                validations.append(
                    CandidateValidation(
                        candidate_id=candidate.id,
                        outcome=AuditOutcome.INVALID_COUNTEREXAMPLE,
                        method="none",
                        premises_hold=False,
                        result_reproduced=False,
                        contradiction_confirmed=False,
                        rationale="Candidate targets a different claim.",
                    )
                )
                continue
            try:
                validation = self._verifier.validate(dossier, claim, candidate)
                if validation.candidate_id != candidate.id:
                    raise ValueError("verifier returned a validation for a different candidate")
                validations.append(validation)
            except Exception as exc:
                # One failed check should not erase other candidates or crash the
                # whole audit. Keep the uncertainty explicit and bounded.
                validations.append(
                    CandidateValidation(
                        candidate_id=candidate.id,
                        outcome=AuditOutcome.UNRESOLVED,
                        method="none",
                        premises_hold=None,
                        result_reproduced=None,
                        contradiction_confirmed=None,
                        rationale=f"Candidate validation failed: {type(exc).__name__}",
                    )
                )

        confirmed = next((item for item in validations if item.confirms_refutation), None)
        if confirmed is not None:
            candidate = next(item for item in candidates if item.id == confirmed.candidate_id)
            return ClaimAuditResult(
                claim_id=claim.id,
                outcome=AuditOutcome.REFUTED,
                attempted=True,
                candidates=candidates,
                validations=tuple(validations),
                finding=(
                    f"Validated counterexample: {candidate.scenario}. "
                    f"Contradiction: {candidate.contradiction}"
                ),
                correction=confirmed.rationale,
            )

        all_invalid = bool(validations) and all(
            item.outcome == AuditOutcome.INVALID_COUNTEREXAMPLE
            or item.premises_hold is False
            or item.contradiction_confirmed is False
            for item in validations
        )
        outcome = (
            AuditOutcome.UNRESOLVED
            if malformed_candidate_count
            else AuditOutcome.NOT_FALSIFIED if all_invalid else AuditOutcome.UNRESOLVED
        )
        findings = []
        if all_invalid:
            findings.append("All candidate counterexamples were invalid; no claim of correctness is implied.")
        else:
            findings.append("Available checks did not resolve whether a counterexample refutes this claim.")
        if malformed_candidate_count:
            findings.append(
                f"Generator returned {malformed_candidate_count} malformed or duplicate candidate(s); "
                "the audit is incomplete."
            )
        return ClaimAuditResult(
            claim_id=claim.id,
            outcome=outcome,
            attempted=True,
            candidates=candidates,
            validations=tuple(validations),
            finding=" ".join(findings),
        )

    def audit_dossier(self, dossier: TaskDossier) -> AuditReport:
        results = tuple(self.audit_claim(dossier, claim) for claim in dossier.claims)
        return AuditReport(
            user_id=dossier.user_id,
            task_id=dossier.task_id,
            dossier_version=dossier.version,
            results=results,
        )
