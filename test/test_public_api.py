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



def test_root_main_exports_cli_entrypoint():
    from main import main
    from examples.study_agent import main as example_main

    assert main is example_main
