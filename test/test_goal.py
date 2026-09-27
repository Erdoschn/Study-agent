from core.goal import GoalMatcher


def test_extract_goals_preserves_explicit_user_subquestions():
    question = "上下文缓存是什么？我要怎么学呢？我现在想用coder agent，是自己做好还是找免费的自己配置呢？"
    goals = GoalMatcher.extract_goals(question)
    assert goals == [
        "上下文缓存是什么",
        "我要怎么学呢",
        "我现在想用coder agent，是自己做好还是找免费的自己配置呢",
    ]


def test_extract_goals_ignores_empty_segments():
    assert GoalMatcher.extract_goals("  ？  ") == []
