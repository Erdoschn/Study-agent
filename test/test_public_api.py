def test_core_exports_learning_apis():
    from core import (
        AssessmentEvaluator,
        GoalMatcher,
        KnowledgeGraph,
        KnowledgeNode,
        LearnerState,
        normalize_difficulty,
    )

    assert AssessmentEvaluator is not None
    assert GoalMatcher is not None
    assert KnowledgeGraph is not None
    assert KnowledgeNode is not None
    assert LearnerState is not None
    assert normalize_difficulty("postgraduate")[0] == "postgraduate"
