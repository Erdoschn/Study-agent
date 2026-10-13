from __future__ import annotations

import pytest

from core.http_security import is_loopback_host, require_auth_for_non_loopback_bind


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.10.20.30", "localhost", "LOCALHOST.", "::1", "[::1]"],
)
def test_loopback_hosts_do_not_require_an_api_key(host):
    assert is_loopback_host(host)
    require_auth_for_non_loopback_bind(
        host,
        "",
        api_name="test API",
        api_key_env="TEST_API_KEY",
        host_env="TEST_API_HOST",
    )


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.20", "study-agent.local", "", "not-an-ip"],
)
def test_non_loopback_or_unknown_hosts_require_an_api_key(host):
    assert not is_loopback_host(host)
    with pytest.raises(RuntimeError, match="TEST_API_KEY"):
        require_auth_for_non_loopback_bind(
            host,
            "",
            api_name="test API",
            api_key_env="TEST_API_KEY",
            host_env="TEST_API_HOST",
        )


def test_non_loopback_bind_accepts_a_configured_api_key():
    require_auth_for_non_loopback_bind(
        "0.0.0.0",
        "long-random-test-key",
        api_name="test API",
        api_key_env="TEST_API_KEY",
        host_env="TEST_API_HOST",
    )


def test_whitespace_only_key_does_not_count_as_configured():
    with pytest.raises(RuntimeError, match="TEST_API_KEY"):
        require_auth_for_non_loopback_bind(
            "192.168.0.10",
            "   ",
            api_name="test API",
            api_key_env="TEST_API_KEY",
            host_env="TEST_API_HOST",
        )


@pytest.mark.parametrize(
    ("module_name", "key_env", "host_env", "marker"),
    [
        ("examples.study_agent_api", "STUDY_AGENT_API_KEY", "STUDY_AGENT_HOST", "load_config"),
        ("examples.coder_web_api", "CODER_WEB_API_KEY", "CODER_WEB_HOST", "WorkspaceFS"),
        (
            "examples.deepseek_web_study_agent_api",
            "STUDY_AGENT_WEB_API_KEY",
            "STUDY_AGENT_WEB_HOST",
            "build_agent",
        ),
    ],
)
def test_api_startup_rejects_non_loopback_bind_without_key_before_side_effects(
    module_name, key_env, host_env, marker, monkeypatch
):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "HOST", "0.0.0.0")
    monkeypatch.setattr(module, "API_KEY", "")

    def unexpected_call(*_args, **_kwargs):
        pytest.fail(f"{marker} must not run before bind authentication is validated")

    if marker == "load_config":
        monkeypatch.setattr(module, "load_config", unexpected_call)
    elif marker == "WorkspaceFS":
        monkeypatch.setattr(module, "WorkspaceFS", unexpected_call)
    else:
        monkeypatch.setattr(module, "build_agent", unexpected_call)

    with pytest.raises(RuntimeError, match=key_env):
        module.main()
