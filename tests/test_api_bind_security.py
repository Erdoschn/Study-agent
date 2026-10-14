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


@pytest.mark.parametrize(
    ("origin", "port", "allowed"),
    [
        ("", 8000, True),  # server-to-server OpenAI-compatible clients
        ("http://127.0.0.1:8000", 8000, True),
        ("http://localhost:8000/", 8000, True),
        ("http://[::1]:8000", 8000, True),
        ("http://127.0.0.1:8001", 8000, False),
        ("https://127.0.0.1:8000", 8000, False),
        ("http://attacker.example:8000", 8000, False),
        ("http://127.0.0.1:8000.evil.example", 8000, False),
        ("http://user@127.0.0.1:8000", 8000, False),
        ("http://127.0.0.1:8000/path", 8000, False),
        ("null", 8000, False),
    ],
)
def test_local_browser_origin_validation(origin, port, allowed):
    from core.http_security import is_allowed_local_browser_origin

    assert is_allowed_local_browser_origin(origin, port) is allowed


def test_read_limited_body_accepts_exact_length_and_rejects_over_limit():
    import io
    from email.message import Message
    from core.http_security import RequestBodyTooLarge, read_limited_request_body

    class FakeHandler:
        def __init__(self, content_length, body):
            self.headers = Message()
            if content_length is not None:
                self.headers["Content-Length"] = str(content_length)
            self.rfile = io.BytesIO(body)

    assert read_limited_request_body(FakeHandler(4, b"data"), 4) == b"data"
    with pytest.raises(RequestBodyTooLarge):
        read_limited_request_body(FakeHandler(5, b"12345"), 4)


def test_read_limited_body_rejects_ambiguous_or_unsupported_framing():
    import io
    from email.message import Message
    from core.http_security import HTTPRequestError, read_limited_request_body

    class FakeHandler:
        def __init__(self, headers, body=b""):
            self.headers = headers
            self.rfile = io.BytesIO(body)

    duplicated = Message()
    duplicated["Content-Length"] = "1"
    duplicated["Content-Length"] = "1"
    with pytest.raises(HTTPRequestError, match="重复的 Content-Length"):
        read_limited_request_body(FakeHandler(duplicated, b"x"), 4)

    chunked = Message()
    chunked["Transfer-Encoding"] = "chunked"
    with pytest.raises(HTTPRequestError, match="不支持 Transfer-Encoding"):
        read_limited_request_body(FakeHandler(chunked, b"1\\r\\nx"), 4)

    malformed = Message()
    malformed["Content-Length"] = "-1"
    with pytest.raises(HTTPRequestError, match="Content-Length 格式无效"):
        read_limited_request_body(FakeHandler(malformed), 4)


def test_browser_facing_apis_reject_untrusted_origins():
    from examples import coder_web_api, deepseek_web_study_agent_api, study_agent_api

    assert coder_web_api._allowed_browser_origin("http://localhost:8002")
    assert not coder_web_api._allowed_browser_origin("https://attacker.example")
    assert study_agent_api._allowed_browser_origin("http://127.0.0.1:8000")
    assert not study_agent_api._allowed_browser_origin("http://attacker.example:8000")
    assert deepseek_web_study_agent_api._allowed_browser_origin("http://localhost:8001")
    assert not deepseek_web_study_agent_api._allowed_browser_origin("http://evil.example:8001")


def test_standalone_deepseek_api_requires_auth_before_browser_startup(monkeypatch):
    from examples import deepseek_web_api

    monkeypatch.setattr(deepseek_web_api, "HOST", "0.0.0.0")
    monkeypatch.setattr(deepseek_web_api, "API_KEY", "")

    def should_not_start(*_args, **_kwargs):
        pytest.fail("browser must not start before non-loopback authentication is checked")

    monkeypatch.setattr(deepseek_web_api, "CoderBrowserSession", should_not_start)
    with pytest.raises(RuntimeError, match="STUDY_AGENT_WEB_API_KEY"):
        deepseek_web_api.main()
