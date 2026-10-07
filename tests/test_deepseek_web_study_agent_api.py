from examples.deepseek_web_study_agent_api import Handler, WEB_ROOT, build_web_only_config, web_asset_path


def test_web_api_extracts_latest_user_message_from_history():
    prompt = Handler._prompt([
        {"role": "system", "content": "teach clearly"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new"},
    ])
    assert "【用户】" in prompt
    assert "new" in prompt
    assert "teach clearly" in prompt


def test_web_api_rejects_missing_user_message():
    try:
        Handler._prompt([{"role": "system", "content": "only system"}])
    except ValueError as exc:
        assert "user" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_web_api_accepts_multimodal_text_content():
    assert Handler._message_text([
        {"type": "input_text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]) == "hello\nworld"


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


def test_web_api_does_not_cooldown_its_only_browser_model():
    config = {"providers": {}, "models": {}}
    isolated = build_web_only_config(config)

    from examples.deepseek_web_study_agent_api import build_agent

    agent = build_agent(config)
    registry = agent.reasoner.model_router.registry

    assert list(isolated["models"]) == ["deepseek-web"]
    assert registry.BASE_COOLDOWN_SECONDS == 0.0
    assert registry.MAX_COOLDOWN_SECONDS == 0.0
    assert registry.PROVIDER_COOLDOWN_SECONDS == 0.0


def test_web_api_supplies_browser_defaults_when_local_config_lacks_them():
    isolated = build_web_only_config({"providers": {}, "models": {}})
    provider = isolated["providers"]["deepseek_web"]
    assert provider["type"] == "browser"
    assert provider["url"] == "https://chat.deepseek.com/"
    assert provider["browser_channel"] == "msedge"
    assert isolated["models"]["deepseek-web"]["paid"] is False


def test_web_api_resolves_only_files_inside_web_root():
    assert web_asset_path("/") == WEB_ROOT / "index.html"
    assert web_asset_path("/index.html") == WEB_ROOT / "index.html"
    assert web_asset_path("/../examples/deepseek_web_study_agent_api.py") is None


def test_web_frontend_does_not_force_markdown_line_breaks():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert "breaks:false" in source
    assert "breaks:true" not in source


def test_web_frontend_exists():
    path = WEB_ROOT / "index.html"
    assert path.is_file()
    assert path.stat().st_size > 1000
