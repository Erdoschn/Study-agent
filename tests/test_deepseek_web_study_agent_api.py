from examples.deepseek_web_study_agent_api import Handler, build_web_only_config


def test_web_api_extracts_latest_user_message():
    assert Handler._prompt([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new"},
    ]) == "new"


def test_web_api_rejects_missing_user_message():
    try:
        Handler._prompt([{"role": "system", "content": "only system"}])
    except ValueError as exc:
        assert "user" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_web_api_isolates_browser_model():
    config = {
        "providers": {
            "deepseek_web": {"type": "browser", "enabled": True},
            "other": {"type": "openai_compatible", "enabled": True},
        },
        "models": {
            "deepseek-web": {"provider": "deepseek_web", "enabled": True, "paid": False},
            "other-model": {"provider": "other", "enabled": True, "paid": False},
        },
    }
    isolated = build_web_only_config(config)
    assert list(isolated["providers"]) == ["deepseek_web"]
    assert list(isolated["models"]) == ["deepseek-web"]


def test_web_api_supplies_browser_defaults_when_local_config_lacks_them():
    isolated = build_web_only_config({"providers": {}, "models": {}})
    provider = isolated["providers"]["deepseek_web"]
    assert provider["type"] == "browser"
    assert provider["url"] == "https://chat.deepseek.com/"
    assert provider["browser_channel"] == "msedge"
    assert isolated["models"]["deepseek-web"]["paid"] is False
