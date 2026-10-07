from examples.deepseek_web_api import Handler


def test_web_api_exposes_only_deepseek_web():
    assert Handler._extract_prompt(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]
    ) == "hello"


def test_web_api_uses_latest_user_message():
    assert Handler._extract_prompt(
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "new"},
        ]
    ) == "new"


def test_web_api_rejects_missing_user_message():
    try:
        Handler._extract_prompt([{"role": "system", "content": "only system"}])
    except ValueError as exc:
        assert "user" in str(exc)
    else:
        raise AssertionError("expected ValueError")
