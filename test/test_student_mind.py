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
