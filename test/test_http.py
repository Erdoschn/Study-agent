import urllib.error

import pytest

from tools.search import HttpClient, HttpRequestError


def test_http_client_success(monkeypatch):
    class FakeResponse:
        status = 200
        reason = "OK"
        headers = {"Content-Type": "text/plain"}

        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = HttpClient(timeout=7, retries=0).get(
        "https://example.com/test",
        headers={"User-Agent": "StudyAgent-Test"},
    )

    assert response.status == 200
    assert response.reason == "OK"
    assert response.body == b"ok"
    assert response.attempts == 1
    assert captured["url"] == "https://example.com/test"
    assert captured["timeout"] == 7


def test_http_client_retryable_http_error(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=30):
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "Server Error",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("tools.search.http.time.sleep", lambda _: None)

    with pytest.raises(HttpRequestError) as exc_info:
        HttpClient(timeout=7, retries=2, backoff_seconds=0).get(
            "https://example.com/test"
        )

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.retryable is True
    assert exc.attempts == 3
    assert len(calls) == 3
    assert "HTTPError" in str(exc)


def test_http_client_non_retryable_http_error(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=30):
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(HttpRequestError) as exc_info:
        HttpClient(timeout=7, retries=2, backoff_seconds=0).get(
            "https://example.com/test"
        )

    exc = exc_info.value
    assert exc.status_code == 404
    assert exc.retryable is False
    assert exc.attempts == 1
    assert len(calls) == 1


def test_http_client_network_error_retries(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=30):
        calls.append(1)
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("tools.search.http.time.sleep", lambda _: None)

    with pytest.raises(HttpRequestError) as exc_info:
        HttpClient(timeout=7, retries=1, backoff_seconds=0).get(
            "https://example.com/test"
        )

    exc = exc_info.value
    assert exc.status_code is None
    assert exc.retryable is True
    assert exc.attempts == 2
    assert len(calls) == 2
    assert "connection reset" in str(exc)
