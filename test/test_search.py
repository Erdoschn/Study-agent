import urllib.error

from tools.search import (
    ArxivSearchProvider,
    HttpClient,
    SearchQuery,
    SearchResult,
    SearchRouter,
    WikipediaSearchProvider,
)


def test_search_query():
    q = SearchQuery(query="transformer")
    assert q.query == "transformer"
    assert q.source == "arxiv"
    assert q.max_results == 10


def test_arxiv_provider_structure():
    provider = ArxivSearchProvider()
    assert provider.name == "arxiv"
    assert len(provider.API_URLS) == 2


def test_wikipedia_provider_structure():
    provider = WikipediaSearchProvider()
    assert provider.name == "wikipedia"
    assert provider.API_URL == "https://en.wikipedia.org/w/api.php"
    assert provider.SUMMARY_URL.startswith("https://en.wikipedia.org/api/rest_v1/")


def test_router():
    router = SearchRouter()
    router.register(ArxivSearchProvider())
    router.register(WikipediaSearchProvider())
    assert "arxiv" in router.available_sources()
    assert "wikipedia" in router.available_sources()


def test_search_result_structure():
    result = SearchResult(
        source="wikipedia",
        source_type="encyclopedia",
        title="Transformer",
        url="https://en.wikipedia.org/wiki/Transformer",
        abstract="Test abstract",
    )
    assert result.source == "wikipedia"
    assert result.source_type == "encyclopedia"
    assert result.title == "Transformer"
    assert result.url.startswith("https://")


def test_arxiv_query_building(monkeypatch):
    provider = ArxivSearchProvider(http_client=HttpClient(timeout=30, retries=0))
    provider._last_request_time = 0
    captured = {}

    class FakeResponse:
        status = 200
        reason = "OK"
        headers = {}

        def read(self):
            return b'''<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <id>http://arxiv.org/abs/1234.5678</id>
                    <title>Test Transformer Paper</title>
                    <summary>Test abstract.</summary>
                    <published>2026-01-01T00:00:00Z</published>
                    <updated>2026-01-02T00:00:00Z</updated>
                    <author><name>Test Author</name></author>
                </entry>
            </feed>'''

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        return FakeResponse()

    monkeypatch.setattr("tools.search.arxiv.urllib.request.urlopen", fake_urlopen)
    results = provider.search(
        SearchQuery(query="transformer", categories=["cs.LG"], max_results=2)
    )

    assert len(results) == 1
    assert results[0].title == "Test Transformer Paper"
    assert results[0].identifier == "1234.5678"
    assert captured["timeout"] == 30
    assert "User-agent" in captured["headers"]
    assert "Accept" in captured["headers"]
    assert "transformer" in captured["url"]
    assert "cat%3Acs.LG" in captured["url"]


def test_arxiv_detailed_response(monkeypatch):
    provider = ArxivSearchProvider(http_client=HttpClient(retries=0))
    provider._last_request_time = 0

    class FakeResponse:
        status = 200
        reason = "OK"
        headers = {}

        def read(self):
            return b'''<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>'''

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(
        "tools.search.arxiv.urllib.request.urlopen",
        lambda request, timeout=30: FakeResponse(),
    )
    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is True
    assert response.provider == "arxiv"
    assert response.error is None
    assert response.results == []
    assert response.metadata["status_code"] == 200
    assert response.metadata["endpoint"] == provider.API_URLS[0]


def test_arxiv_406_fallback(monkeypatch):
    provider = ArxivSearchProvider(http_client=HttpClient(retries=0))
    provider._last_request_time = 0
    calls = []

    class FakeResponse:
        status = 200
        reason = "OK"
        headers = {}

        def read(self):
            return b'''<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>'''

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def fake_urlopen(request, timeout=30):
        calls.append(request.full_url.split("?", 1)[0])
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 406, "Not Acceptable", hdrs=None, fp=None
            )
        return FakeResponse()

    monkeypatch.setattr("tools.search.arxiv.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("tools.search.arxiv.time.sleep", lambda _: None)
    assert provider.search(SearchQuery(query="transformer")) == []
    assert calls == list(provider.API_URLS)


def test_arxiv_non_406_error(monkeypatch):
    provider = ArxivSearchProvider(http_client=HttpClient(retries=0))
    provider._last_request_time = 0

    def fake_urlopen(request, timeout=30):
        raise urllib.error.HTTPError(
            request.full_url, 500, "Server Error", hdrs=None, fp=None
        )

    monkeypatch.setattr("tools.search.arxiv.urllib.request.urlopen", fake_urlopen)
    try:
        provider.search(SearchQuery(query="transformer"))
    except RuntimeError as exc:
        assert "HTTPError" in str(exc)
        assert "500" in str(exc)
        return
    raise AssertionError("非 406 HTTP 错误应该抛出 RuntimeError")


def test_arxiv_parse_error(monkeypatch):
    provider = ArxivSearchProvider(http_client=HttpClient(retries=0))
    provider._last_request_time = 0

    class FakeResponse:
        status = 200
        reason = "OK"
        headers = {}

        def read(self):
            return b"not xml"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(
        "tools.search.arxiv.urllib.request.urlopen",
        lambda request, timeout=30: FakeResponse(),
    )
    try:
        provider.search(SearchQuery(query="transformer"))
    except RuntimeError as exc:
        assert "XML 解析失败" in str(exc)
        return
    raise AssertionError("非法 XML 应该抛出 RuntimeError")


def test_wikipedia_search_uses_shared_http_client():
    class FakeHttpClient:
        def __init__(self):
            self.calls = []

        def get(self, url, *, headers=None):
            self.calls.append((url, headers))

            class Response:
                status = 200
                reason = "OK"
                headers = {"Content-Type": "application/json"}
                attempts = 1

                def __init__(self, body):
                    self.body = body

            if "w/api.php" in url:
                body = b'["transformer",["Transformer"],[""],["https://en.wikipedia.org/wiki/Transformer"]]'
            else:
                body = b'{"title":"Transformer","extract":"Attention is a mechanism.","wikibase_item":"Q123","timestamp":"2026-01-01T00:00:00Z","content_urls":{"desktop":{"page":"https://en.wikipedia.org/wiki/Transformer"}}}'
            return Response(body)

    client = FakeHttpClient()
    provider = WikipediaSearchProvider(http_client=client)
    provider._last_request_time = 0

    results = provider.search(SearchQuery(query="transformer", max_results=1))

    assert len(results) == 1
    assert results[0].title == "Transformer"
    assert results[0].identifier == "Q123"
    assert len(client.calls) == 2
    assert all(call[1]["Accept"] == "application/json" for call in client.calls)


def test_wikipedia_detailed_response_http_error():
    class FakeHttpClient:
        def get(self, url, *, headers=None):
            from tools.search import HttpRequestError
            raise HttpRequestError(
                "HTTPError: HTTP Error 503: Service Unavailable",
                status_code=503,
                reason="Service Unavailable",
                retryable=True,
                attempts=3,
            )

    provider = WikipediaSearchProvider(
        http_client=FakeHttpClient()
    )
    provider._last_request_time = 0
    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "http"
    assert response.error.status_code == 503
    assert response.error.retryable is True
    assert response.error.attempts == 3


def test_wikipedia_json_parse_error():
    class FakeHttpClient:
        def get(self, url, *, headers=None):
            class Response:
                status = 200
                reason = "OK"
                headers = {}
                body = b"not json"
                attempts = 1
            return Response()

    provider = WikipediaSearchProvider(http_client=FakeHttpClient())
    provider._last_request_time = 0
    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "parse"
    assert "JSON 解析失败" in response.error.message


def test_empty_query():
    provider = WikipediaSearchProvider()
    try:
        provider.search(SearchQuery(query=""))
    except ValueError:
        return
    raise AssertionError("空查询应该被拒绝")
