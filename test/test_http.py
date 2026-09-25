import pytest
import requests

from tools.search import HttpClient, HttpRequestError, SearchTimeoutError


class FakeResponse:
    def __init__(self, status_code=200, reason="OK", content=b"ok", headers=None):
        self.status_code = status_code
        self.reason = reason
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, responses=None, exceptions=None):
        self.responses = list(responses or [])
        self.exceptions = list(exceptions or [])
        self.calls = []

    def get(self, url, *, headers=None, timeout=None):
        self.calls.append((url, headers, timeout))
        if self.exceptions:
            raise self.exceptions.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse()


def test_http_client_success():
    session = FakeSession(
        responses=[
            FakeResponse(
                status_code=200,
                reason="OK",
                content=b"ok",
                headers={"Content-Type": "text/plain"},
            )
        ]
    )

    response = HttpClient(
        timeout=7,
        retries=0,
        session=session,
    ).get(
        "https://example.com/test",
        headers={"User-Agent": "StudyAgent-Test"},
    )

    assert response.status == 200
    assert response.reason == "OK"
    assert response.body == b"ok"
    assert response.attempts == 1
    assert session.calls[0][0] == "https://example.com/test"
    assert session.calls[0][2] == 7
    assert session.calls[0][1]["User-Agent"] == "StudyAgent-Test"


def test_http_client_retryable_http_error(monkeypatch):
    session = FakeSession(
        responses=[
            FakeResponse(status_code=500, reason="Server Error", content=b"error"),
            FakeResponse(status_code=500, reason="Server Error", content=b"error"),
            FakeResponse(status_code=500, reason="Server Error", content=b"error"),
        ]
    )
    monkeypatch.setattr("tools.search.http.time.sleep", lambda _: None)

    with pytest.raises(HttpRequestError) as exc_info:
        HttpClient(
            timeout=7,
            retries=2,
            backoff_seconds=0,
            session=session,
        ).get("https://example.com/test")

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.retryable is True
    assert exc.attempts == 3
    assert len(session.calls) == 3
    assert "HTTPError" in str(exc)


def test_http_client_non_retryable_http_error():
    session = FakeSession(
        responses=[FakeResponse(status_code=404, reason="Not Found", content=b"missing")]
    )

    with pytest.raises(HttpRequestError) as exc_info:
        HttpClient(
            timeout=7,
            retries=2,
            backoff_seconds=0,
            session=session,
        ).get("https://example.com/test")

    exc = exc_info.value
    assert exc.status_code == 404
    assert exc.retryable is False
    assert exc.attempts == 1
    assert len(session.calls) == 1


def test_http_client_network_error_retries(monkeypatch):
    session = FakeSession(
        exceptions=[
            requests.exceptions.ConnectionError("connection reset"),
            requests.exceptions.ConnectionError("connection reset"),
        ]
    )
    monkeypatch.setattr("tools.search.http.time.sleep", lambda _: None)

    with pytest.raises(HttpRequestError) as exc_info:
        HttpClient(
            timeout=7,
            retries=1,
            backoff_seconds=0,
            session=session,
        ).get("https://example.com/test")

    exc = exc_info.value
    assert exc.status_code is None
    assert exc.retryable is True
    assert exc.attempts == 2
    assert len(session.calls) == 2
    assert "connection reset" in str(exc)


def test_http_client_timeout_is_explicit(monkeypatch):
    session = FakeSession(
        exceptions=[
            requests.exceptions.Timeout("read timed out"),
            requests.exceptions.Timeout("read timed out"),
        ]
    )
    monkeypatch.setattr("tools.search.http.time.sleep", lambda _: None)

    with pytest.raises(SearchTimeoutError) as exc_info:
        HttpClient(
            timeout=3,
            retries=1,
            backoff_seconds=0,
            session=session,
        ).get("https://example.com/slow")

    assert "SEARCH_TIMEOUT" in str(exc_info.value)
    assert "timeout=3s" in str(exc_info.value)
    assert len(session.calls) == 2
