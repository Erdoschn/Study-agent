import requests

from tools.search import (
    ArxivSearchProvider,
    HttpClient,
    HttpRequestError,
    SearchQuery,
    SearchResult,
    SearchRouter,
    SearchTimeoutError,
    WikipediaSearchProvider,
)


ATOM = b'''<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>https://arxiv.org/abs/1234.5678</id>
        <title>Test Transformer Paper</title>
        <summary>Test abstract.</summary>
        <published>2026-01-01T00:00:00Z</published>
        <updated>2026-01-02T00:00:00Z</updated>
        <author><name>Test Author</name></author>
        <link href="https://arxiv.org/html/1234.5678" rel="alternate" type="text/html" />
    </entry>
</feed>'''


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, headers=None):
        self.calls.append((url, headers))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeResponse:
    def __init__(self, status=200, reason="OK", body=ATOM, attempts=1):
        self.status = status
        self.reason = reason
        self.body = body
        self.attempts = attempts
        self.headers = {"Content-Type": "application/atom+xml"}


def test_search_query():
    q = SearchQuery(query="transformer")
    assert q.query == "transformer"
    assert q.source == "arxiv"
    assert q.max_results == 10


def test_arxiv_provider_structure():
    provider = ArxivSearchProvider()
    assert provider.name == "arxiv"
    assert len(provider.API_URLS) == 2
    assert len(provider.endpoints) == 4
    assert provider.endpoints[0] == "http://export.arxiv.org/api/query"


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


def test_arxiv_query_building():
    assert (
        ArxivSearchProvider._build_search_query("transformer", ["cs.LG"])
        == "all:transformer AND (cat:cs.LG)"
    )
    assert (
        ArxivSearchProvider._build_search_query(
            '"multi-head attention"',
            ["cs.LG", "cs.CL"],
        )
        == 'all:"multi-head attention" AND (cat:cs.LG OR cat:cs.CL)'
    )


def test_arxiv_request_and_parse():
    client = FakeHttpClient([FakeResponse()])
    provider = ArxivSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(
        SearchQuery(query="transformer", categories=["cs.LG"], max_results=2)
    )

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].title == "Test Transformer Paper"
    assert response.results[0].identifier == "1234.5678"
    assert response.results[0].authors == ["Test Author"]
    assert response.results[0].url == "https://arxiv.org/html/1234.5678"
    assert response.metadata["transport"] == "requests"
    assert response.metadata["endpoint"] == provider.endpoints[0]
    assert "all%3Atransformer" in client.calls[0][0]
    assert "cat%3Acs.LG" in client.calls[0][0]
    assert client.calls[0][1]["Accept"].startswith("application/atom+xml")


def test_arxiv_406_fallback():
    client = FakeHttpClient(
        [
            HttpRequestError(
                "HTTPError: HTTP Error 406: Not Acceptable",
                status_code=406,
                reason="Not Acceptable",
                response_body="",
                retryable=False,
                attempts=1,
            ),
            FakeResponse(),
        ]
    )
    provider = ArxivSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is True
    assert len(client.calls) == 2
    assert client.calls[0][0].startswith(provider.endpoints[0])
    assert client.calls[1][0].startswith(provider.endpoints[1])


def test_arxiv_network_fallback():
    client = FakeHttpClient(
        [
            requests.exceptions.ConnectionError("connection reset"),
            FakeResponse(),
        ]
    )
    provider = ArxivSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is True
    assert len(client.calls) == 2


def test_arxiv_all_endpoints_fail_structured():
    client = FakeHttpClient(
        [
            HttpRequestError(
                "HTTPError: HTTP Error 406: Not Acceptable",
                status_code=406,
                reason="Not Acceptable",
                attempts=1,
            )
            for _ in range(4)
        ]
    )
    provider = ArxivSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "http"
    assert response.error.status_code == 406
    assert len(client.calls) == 4


def test_arxiv_non_fallback_http_error():
    client = FakeHttpClient(
        [
            HttpRequestError(
                "HTTPError: HTTP Error 400: Bad Request",
                status_code=400,
                reason="Bad Request",
                attempts=1,
            )
        ]
    )
    provider = ArxivSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "http"
    assert response.error.status_code == 400
    assert len(client.calls) == 1


def test_arxiv_parse_error():
    client = FakeHttpClient([FakeResponse(body=b"not xml")])
    provider = ArxivSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "parse"
    assert "XML 解析失败" in response.error.message


def test_arxiv_invalid_query():
    provider = ArxivSearchProvider()
    response = provider.search_detailed(SearchQuery(query=""))
    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "validation"


def test_wikipedia_search_uses_shared_http_client():
    class FakeWikipediaHttpClient:
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

            if "list=search" in url:
                body = b'{"query":{"search":[{"pageid":123,"title":"Transformer","snippet":"Attention is a mechanism.","timestamp":"2026-01-01T00:00:00Z"}]}}'
            else:
                body = b'{"query":{"pages":{"123":{"pageid":123,"title":"Transformer","extract":"Attention is a mechanism.","fullurl":"https://en.wikipedia.org/wiki/Transformer","timestamp":"2026-01-01T00:00:00Z"}}}}'
            return Response(body)

    client = FakeWikipediaHttpClient()
    provider = WikipediaSearchProvider(http_client=client)
    provider._last_request_time = 0

    results = provider.search(SearchQuery(query="transformer", max_results=1))

    assert len(results) == 1
    assert results[0].title == "Transformer"
    assert results[0].identifier == "123"
    assert len(client.calls) == 2
    assert "list=search" in client.calls[0][0]
    assert "srsearch=transformer" in client.calls[0][0]
    assert "srlimit=1" in client.calls[0][0]
    assert all(call[1]["Accept"] == "application/json" for call in client.calls)


def test_wikipedia_detailed_response_http_error():
    class FakeWikipediaHttpClient:
        def get(self, url, *, headers=None):
            raise HttpRequestError(
                "HTTPError: HTTP Error 503: Service Unavailable",
                status_code=503,
                reason="Service Unavailable",
                retryable=True,
                attempts=3,
            )

    provider = WikipediaSearchProvider(http_client=FakeWikipediaHttpClient())
    provider._last_request_time = 0
    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "http"
    assert response.error.status_code == 503
    assert response.error.retryable is True
    assert response.error.attempts == 3


def test_wikipedia_json_parse_error():
    class FakeWikipediaHttpClient:
        def get(self, url, *, headers=None):
            class Response:
                status = 200
                reason = "OK"
                headers = {}
                body = b"not json"
                attempts = 1
            return Response()

    provider = WikipediaSearchProvider(http_client=FakeWikipediaHttpClient())
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


def test_wikipedia_fulltext_search_returns_multiple_ranked_candidates():
    class FakeWikipediaHttpClient:
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

            if "list=search" in url:
                body = (
                    b'{"query":{"search":['
                    b'{"pageid":1,"title":"Attention","snippet":"...","timestamp":"2026-01-01T00:00:00Z"},'
                    b'{"pageid":2,"title":"Attention mechanism","snippet":"...","timestamp":"2026-02-01T00:00:00Z"}'
                    b']}}'
                )
            else:
                body = (
                    b'{"query":{"pages":{'
                    b'"1":{"pageid":1,"title":"Attention","extract":"First result.","fullurl":"https://en.wikipedia.org/wiki/Attention"},'
                    b'"2":{"pageid":2,"title":"Attention mechanism","extract":"Second result.","fullurl":"https://en.wikipedia.org/wiki/Attention_mechanism"}'
                    b'}}}'
                )
            return Response(body)

    client = FakeWikipediaHttpClient()
    provider = WikipediaSearchProvider(http_client=client)
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    results = provider.search(SearchQuery(query="attention", max_results=2))

    assert [r.title for r in results] == ["Attention", "Attention mechanism"]
    assert len(client.calls) == 2
    assert "srlimit=2" in client.calls[0][0]


def test_arxiv_timeout_falls_back_to_next_endpoint(monkeypatch):
    class TimeoutHttp:
        def __init__(self):
            self.calls = []
        def get(self, url, *, headers=None):
            self.calls.append(url)
            if len(self.calls) == 1:
                raise SearchTimeoutError("SEARCH_TIMEOUT")
            return FakeResponse()

    http = TimeoutHttp()
    provider = ArxivSearchProvider(http_client=http)
    monkeypatch.setattr(provider, "_wait_for_rate_limit", lambda: None)
    monkeypatch.setattr(provider, "endpoints", property(lambda: ("https://one.test", "https://two.test")))

    response = provider.search_detailed(SearchQuery(query="transformer", source="arxiv"))

    assert response.success is True
    assert len(http.calls) == 2
    assert "one.test" in http.calls[0]
    assert "two.test" in http.calls[1]


def test_search_router_auto_continues_after_timeout():
    class TimeoutProvider:
        name = "slow"
        def search(self, query):
            raise SearchTimeoutError("SEARCH_TIMEOUT")

    class GoodProvider:
        name = "good"
        def search(self, query):
            return [SearchResult(source="good", title="ok", url="https://example.com")]

    router = SearchRouter()
    router.register(TimeoutProvider())
    router.register(GoodProvider())

    results = router.search(
        SearchQuery(
            query="transformer",
            source="auto",
            source_preferences=["slow", "good"],
        )
    )

    assert len(results) == 1
    assert results[0].source == "good"
