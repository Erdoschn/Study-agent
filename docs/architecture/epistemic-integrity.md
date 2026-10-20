# Epistemic Integrity: Detail Preservation and Counterexample-Driven Auditing

## Purpose

Study Agent is a learning aid, not an oracle. It must reduce avoidable errors, expose unresolved uncertainty and preserve the exact task the student asked about. It must never equate a fluent answer, a second model's approval, or failure to find a counterexample with proof of correctness.

This design has two complementary mechanisms:

1. **Task Dossier:** preserve source facts and requirements across prompt compression and role handoffs.
2. **Counterexample-driven audit:** try to falsify material claims, then independently validate each candidate counterexample and produce reproducible feedback.

These are architecture contracts for future implementation. The current `Teacher` and `TeachingValidator` do not yet implement the complete design.

## 1. Task Dossier — one source of truth for the task

Create a versioned, user-scoped task artifact. The Dossier is separate from chat history and from any model-generated summary.

### Required sections

- **Task identity:** `user_id`, `task_id`, `dossier_version`, creation time and lifecycle status.
- **Original sources:** source IDs, content hashes, original file/page/cell/line/URL references, extraction version and provenance. Store source bytes only where appropriate under existing retention and access-control rules.
- **Requirements ledger:** each requirement has a stable ID, exact source span, normalized interpretation, required/optional status and verification state.
- **Given facts and constraints:** values, units, definitions, assumptions, domain restrictions, quantifiers, boundary conditions, diagrams/figures and time/version constraints; each item points to its source.
- **Open questions:** missing, ambiguous, conflicting or unreadable source details. An unknown value stays unknown rather than being filled by model inference.
- **Claim and derivation ledger:** answer claims link to requirements, facts, definitions, derivation steps, tools and evidence.
- **Coverage ledger:** which requirements were addressed, omitted, contradicted or not applicable in the answer, with reasons.
- **Artifacts and audit history:** answer versions, audit packets, candidate counterexamples, verified findings and revision links.

### Non-negotiable invariants

1. Original source material is the authority for what the prompt/file said. A summary is a projection, never a replacement.
2. Facts inferred by a model are marked as inferences, not silently promoted to source facts.
3. Every prompt view records its source Dossier version and the IDs of included facts/requirements. Prompt truncation is observable.
4. If a prompt budget cannot fit all critical facts, pass a structured critical-facts/requirements view and explicit source references. Do not silently drop constraints.
5. A fact or requirement that is missing, conflicting or not extracted must be marked unresolved and surfaced where it affects the answer.
6. A source or task edit creates a new version and invalidates dependent verification results.
7. Dossier records, attachments and knowledge state are always isolated by user and task scope. Apply access control, retention limits and redaction to audit artifacts too.

### Prompt projection, not transcript stuffing

Do not send the full conversation blindly to every role. Build a role-specific view from the same Dossier version:

- **Reasoner view:** goal, required outputs, critical givens, open questions and relevant state.
- **Teacher view:** the requirements checklist, relevant established facts, the draft and selected evidence/derivations.
- **Falsifier view:** exact claim, its stated scope/assumptions, relevant source facts and independent evidence—not the generator's rationale as authority.
- **Verifier view:** candidate counterexample, original claim/specification, source references and independently checkable computations/tests.
- **User-facing audit view:** only the findings, confidence scope, evidence, repair and unresolved questions.

Every view includes IDs and references so details can be recovered. Character limits remain valid for transport safety, but truncation must be recorded; critical facts cannot be silently removed.

## 2. Answer claim ledger

Before auditing, decompose the candidate answer into atomic, testable claims. Each claim should contain:

- stable `claim_id` and exact answer span;
- normalized claim, including quantifiers and scope;
- type: definition, mathematical statement, calculation, empirical/current fact, causal claim, code/API behavior, analogy or illustrative example;
- required assumptions and boundary conditions;
- linked Dossier facts and evidence references;
- materiality/risk and whether it is central to the learning objective;
- verification method appropriate to the claim type.

Do not pretend all prose is equally verifiable. Distinguish factual propositions from analogies, pedagogical simplifications, recommendations and value judgements. An analogy can be useful while still needing an explicit boundary where it stops applying.

## 3. Counterexample-driven audit

The auditor must not merely ask, "Is this answer correct?" Its first job is to attempt to falsify each material claim within the stated assumptions.

### Step A — generate an adversarial challenge

For each high-risk or central claim, ask a Falsifier role to search for:

- boundary cases, degenerate values, limits, counter-assumptions and domain restrictions;
- missing necessary conditions or an invalid implication/quantifier;
- a concrete case satisfying the original task's givens but contradicting the claim;
- conflicting primary/authoritative sources, versions or dates;
- an alternative calculation, independent derivation, or failing property/test;
- cases where an analogy or simplified explanation is mistaken for a literal rule.

The falsifier should be encouraged to find a real contradiction, not to manufacture one. It must be allowed to return "no plausible counterexample found" or "not falsifiable by the available method."

### Step B — record a candidate counterexample as data

A candidate is not a finding merely because a model invented it. Use a structured record:

```json
{
  "target_claim_id": "claim-17",
  "candidate_id": "counterexample-3",
  "scenario": "Concrete input / case / source passage",
  "premises_satisfied": ["condition-1", "condition-2"],
  "predicted_result": "What follows in this case",
  "contradiction": "Why this conflicts with the target claim",
  "evidence_refs": ["source-5#page=3"],
  "reproduction": {
    "method": "symbolic_check | calculation | test | primary_source",
    "input": {},
    "expected": "",
    "observed": ""
  }
}
```

The example is illustrative schema, not a requirement that all domains use the same reproduction tool.

### Step C — independently validate the counterexample

A separate verifier checks, in this order:

1. Does the proposed case actually satisfy the original problem, claim assumptions and domain constraints?
2. Are all input values, definitions, units, versions and source citations real and relevant?
3. Does a deterministic calculation, executable test, formal checker or authoritative source reproduce the stated result?
4. Does that result logically contradict the exact target claim, rather than a stronger or distorted paraphrase?
5. Is the counterexample relevant to the user's task, or merely an out-of-scope edge case?

Only a validated contradiction can be promoted to a confirmed error. The verifier must be free to reject an invalid counterexample. The original candidate answer and falsifier rationale are data to check, not authorities.

### Step D — classify the result without overclaiming

Use explicit epistemic outcomes:

- `REFUTED`: a valid counterexample or decisive contradictory evidence disproves the claim as stated.
- `SUPPORTED_WITH_EVIDENCE`: independent evidence/calculation supports the claim within a recorded scope. This is not an absolute guarantee.
- `NOT_FALSIFIED`: attempted challenges did not produce a validated counterexample. This does **not** mean the claim is proven true.
- `UNRESOLVED`: sources conflict, required details are missing, or the available method cannot decide.
- `NOT_TESTED`: this claim was outside the selected audit scope.
- `INVALID_COUNTEREXAMPLE`: a generated candidate did not satisfy the premises or did not contradict the claim.

Overall answer status is derived from the claim ledger, requirement coverage and verified findings. A single generic model response of `PASS` must not set an unqualified `answer_verified=true` flag.

## 4. Match verification method to claim type

| Claim type | Preferred falsification/verification | Important limitation |
|---|---|---|
| Arithmetic, units, numerical result | Recompute independently; check dimensions, sign, rounding and substitutions | Correct arithmetic cannot repair a misread premise |
| Mathematical identity/theorem | Boundary cases, quantifier/domain checks, symbolic checks; formal proof checker where available | Sampled tests can disprove but generally cannot prove a universal theorem |
| Physics/science model | Inspect assumptions and limiting cases; recompute equations; consult authoritative references | A model can be valid only in its declared regime |
| Current factual claim | Retrieve primary/authoritative sources; verify date, version and context; search for contradictory evidence | No retrieved contradiction is not proof; sources may be incomplete or correlated |
| Code/API behavior | Run tests, generated edge cases and property/metamorphic tests in a bounded sandbox; check documented API versions | Passing tests only establish behavior over tested inputs |
| Conceptual explanation/analogy | Search for a counter-case; state scope and where the analogy breaks; contrast with formal definition | Some explanations are useful models, not literal universal propositions |
| Contested interpretation/value judgement | Present competing well-sourced interpretations and unresolved disagreement | Do not force a binary true/false verdict where evidence cannot justify one |

Use available deterministic tools or source material before adding another model call. A second model or model family can provide useful challenge diversity, but model diversity alone is not independent evidence.

## 5. Useful audit feedback

An audit finding is useful only if a student or Teacher can act on it. Each finding should include:

- target claim and exact location in the answer;
- outcome and calibrated severity;
- concrete counterexample or contradictory evidence;
- why the counterexample satisfies the original assumptions;
- reproducible calculation/test or source location;
- minimal correction, missing condition or qualification;
- whether the finding is confirmed, suspected or unresolved.

Avoid ungrounded feedback such as "the explanation may be inaccurate" with no claim, condition, evidence or correction.

After revision, re-run the tests associated with affected claims and re-check the Dossier requirement ledger. Keep the original, revised answer and audit packets linked by version. Do not overwrite verified source facts while revising prose.

## 6. Layer ownership

- **Domain:** Dossier entities, claim/requirement types, evidence and audit outcome semantics.
- **Application:** lifecycle and orchestration: dossier build → answer → claim extraction → falsification → candidate validation → repair → regression audit.
- **Reasoning:** generate claims and candidate counterexamples; it may not mark its own candidate as verified.
- **Ports:** source retrieval, deterministic calculators/checkers, test runners, model roles, and audit/event sinks.
- **Infrastructure:** file parsing, reference retrieval, math/code verification tools, provider adapters, storage and telemetry.
- **Interfaces:** communicate scope and uncertainty accurately; never convert "not falsified" to "verified true".

The Falsifier, Verifier and Teacher are logical roles, not necessarily three separate paid model calls on every request. Route verification depth by task risk, claim uncertainty, availability of deterministic checks and expected impact. Never skip required checks merely because a model is confident.

## 7. Evaluation and regression tests

Measure reliability by behavior, not by the number of critic calls.

Required evaluation slices:

- detail retention: preserve givens, units, assumptions, requested output constraints and sub-questions after context compaction;
- traceability: every critical answer claim links to source facts, derivations or evidence;
- adversarial recall: proportion of known incorrect claims for which the system finds and validates a real counterexample;
- counterexample precision: proportion of accepted counterexamples that actually satisfy premises and contradict the claim;
- false-positive rate: valid claims wrongly rejected by fabricated/out-of-scope counterexamples;
- calibration: distinguish `REFUTED`, `SUPPORTED_WITH_EVIDENCE`, `NOT_FALSIFIED` and `UNRESOLVED` correctly;
- revision quality: confirmed error fixed without dropping other requirements or introducing new errors;
- audit usefulness: feedback is actionable, reproducible and tied to a source or check.

Fixtures should include misleading but plausible candidate answers, omitted assumptions, wrong units/signs, boundary cases, a valid claim for which all generated counterexamples are invalid, conflicting sources, unavailable verifier tools, and prompts that exceed the context budget.

## 8. Product promise

Study Agent should promise **traceable, challenge-tested assistance with explicit uncertainty**, not absolute correctness. Verification results are scoped to the sources, assumptions, tools and tests actually used. When the available process cannot decide, it must say what remains unresolved and how to check it.
