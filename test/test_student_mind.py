from core.reasoner import AgentReasoner
from core.state import StudentState, StudentMind


def test_student_mind_tracks_short_and_long_term_bdi():
    student = StudentState()
    student.apply_mind_update({
        "short_term": {
            "beliefs": ["QK^T measures pairwise compatibility"],
            "desires": ["understand attention"],
            "intentions": ["derive the dimensions by hand"],
        },
        "long_term": {
            "beliefs": ["I learn better when I derive before coding"],
            "desires": ["build a reliable study agent"],
            "intentions": ["test each subsystem"],
        },
        "recent_decisions": ["derive QK^T before asking for code"],
    })

    assert student.mind.short_term.beliefs == ["QK^T measures pairwise compatibility"]
    assert student.mind.short_term.desires == ["understand attention"]
    assert student.mind.short_term.intentions == ["derive the dimensions by hand"]
    assert student.mind.long_term.desires == ["build a reliable study agent"]
    assert student.mind.recent_decisions == ["derive QK^T before asking for code"]


def test_student_mind_deduplicates_and_bounds_recent_memory():
    mind = StudentMind()
    mind.apply_update({
        "short_term": {"beliefs": [str(i) for i in range(20)]},
        "recent_decisions": [str(i) for i in range(20)],
    })

    assert len(mind.short_term.beliefs) == 8
    assert len(mind.recent_decisions) == 12
    assert mind.short_term.beliefs[-1] == "19"
    assert mind.recent_decisions[-1] == "19"


def test_reasoner_sanitizes_invalid_student_model_update():
    normalized = AgentReasoner._normalize_student_model_update({
        "short_term": {
            "beliefs": ["a", "a", "", 1],
            "desires": "not-a-list",
            "intentions": ["plan"],
        },
        "long_term": {
            "beliefs": ["stable"],
        },
        "recent_decisions": "not-a-list",
    })

    assert normalized["short_term"]["beliefs"] == ["a", "1"]
    assert normalized["short_term"]["desires"] == []
    assert normalized["short_term"]["intentions"] == ["plan"]
    assert normalized["long_term"]["beliefs"] == ["stable"]
    assert normalized["recent_decisions"] == []


def test_reasoner_strips_think_block_before_json_parse():
    raw = '<think>private reasoning should not leak</think>\\n{"action":"ANSWER","reasoning_summary":"ok","answer":"done"}'
    decision = AgentReasoner._parse(raw)
    assert decision.action == "ANSWER"
    assert decision.answer == "done"

def test_student_mind_revises_conflicting_belief_with_audit_trail():
    mind = StudentMind()
    mind.long_term.beliefs.append("attention uses one shared score for every head")

    mind.revise_beliefs([{
        "old": "attention uses one shared score for every head",
        "new": "each attention head has its own projected Q K V parameters",
        "horizon": "long_term",
        "status": "REVISED",
        "reason": "verified from the model definition",
    }])

    assert mind.long_term.beliefs == ["each attention head has its own projected Q K V parameters"]
    assert mind.belief_history[-1]["status"] == "REVISED"
    assert mind.belief_history[-1]["old"].startswith("attention uses")


def test_reasoner_normalizes_belief_revisions_safely():
    revisions = AgentReasoner._normalize_belief_revisions([
        {"old": "A", "new": "B", "horizon": "long_term", "status": "revised", "reason": "evidence"},
        {"old": "", "new": "C", "horizon": "short_term", "status": "REVISED"},
        {"old": "D", "new": "E", "horizon": "bad", "status": "REVISED"},
    ])

    assert revisions == [{
        "old": "A", "new": "B", "horizon": "long_term",
        "status": "REVISED", "reason": "evidence",
    }]

def test_student_mind_binds_belief_to_validated_evidence():
    mind = StudentMind()
    evidence = [{
        "source": "wikipedia",
        "title": "Attention mechanism",
        "identifier": "1",
        "harness_relevance": "DIRECT",
    }]
    mind.revise_beliefs([{
        "old": "attention is one shared score",
        "new": "attention uses pairwise compatibility scores",
        "horizon": "long_term",
        "status": "REVISED",
        "reason": "supported by retrieved evidence",
        "evidence_refs": [0, 9, -1, "bad"],
    }], evidence)

    assert mind.belief_support["attention uses pairwise compatibility scores"] == [{
        "index": 0,
        "source": "wikipedia",
        "title": "Attention mechanism",
        "identifier": "1",
        "harness_relevance": "DIRECT",
    }]
    assert mind.belief_history[-1]["evidence_refs"][0]["index"] == 0


def test_student_mind_ignores_invalid_evidence_reference():
    mind = StudentMind()
    evidence = [{"source": "arxiv", "title": "A", "identifier": "1"}]
    mind.revise_beliefs([{
        "old": "A",
        "new": "B",
        "status": "REVISED",
        "evidence_refs": [3],
    }], evidence)

    assert mind.belief_support["B"] == []
    assert mind.belief_history[-1]["evidence_refs"] == []



def test_student_mind_rejects_irrelevant_evidence_as_belief_support():
    from core.state import StudentMind

    mind = StudentMind()
    evidence = [{
        "source": "wikipedia",
        "title": "Unrelated",
        "identifier": "1",
        "harness_relevance": "IRRELEVANT",
    }]
    mind.revise_beliefs([{
        "old": "old belief",
        "new": "new belief",
        "horizon": "long_term",
        "status": "REVISED",
        "reason": "claimed support",
        "evidence_refs": [0],
    }], evidence)

    assert mind.belief_support["new belief"] == []
    assert mind.belief_history[-1]["evidence_refs"] == []
