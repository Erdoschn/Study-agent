import arxiv as arxiv_api
import requests

from tools.search import (
    ArxivSearchProvider,
    SearchQuery,
    SearchResult,
    SearchRouter,
    WikipediaSearchProvider,
)


class FakeAuthor:
    def __init__(self, name: str):
        self.name = name


class FakeResult:
    entry_id = "https://arxiv.org/abs/1234.5678"
    title = "Test Transformer Paper"
    summary = "Test abstract."
    authors = [FakeAuthor("Test Author")]
    published = None
    updated = None

    def get_short_id(self):
        return "1234.5678"


class FakeSearch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeClient:
    def __init__(self, endpoint, results=None, error=None):
        self.endpoint = endpoint
        self._results = results or []
        self._error = error

    def results(self, search):
        if self._error is not None:
            raise self._error
        yield from self._results


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
    assert provider.endpoints[0] == provider.FALLBACK_API_URLS[0]


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


def test_arxiv_api_search_building():
    provider = ArxivSearchProvider()
    search = provider._build_api_search(
        'all:"transformer"',
        SearchQuery(query="transformer", max_results=2),
    )
    assert search.query == 'all:"transformer"'
    assert search.max_results == 2
    assert search.sort_by == arxiv_api.SortCriterion.Relevance
    assert search.sort_order == arxiv_api.SortOrder.Descending


def test_arxiv_search_uses_arxiv_client():
    calls = []

    def factory(endpoint, max_results):
        calls.append((endpoint, max_results))
        return FakeClient(endpoint, results=[FakeResult()])

    provider = ArxivSearchProvider(
        client_factory=factory,
        search_factory=FakeSearch,
    )
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(
        SearchQuery(query="transformer", max_results=2)
    )

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].title == "Test Transformer Paper"
    assert response.results[0].identifier == "1234.5678"
    assert response.metadata["transport"] == "arxiv.py + requests"
    assert response.metadata["endpoint"] == provider.endpoints[0]
    assert calls == [(provider.endpoints[0], 2)]


def test_arxiv_406_fallback():
    calls = []

    def factory(endpoint, max_results):
        calls.append(endpoint)
        if len(calls) == 1:
            return FakeClient(
                endpoint,
                error=arxiv_api.HTTPError(
                    f"{endpoint}?q=test",
                    0,
                    406,
                ),
            )
        return FakeClient(endpoint, results=[FakeResult()])

    provider = ArxivSearchProvider(
        client_factory=factory,
        search_factory=FakeSearch,
    )
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is True
    assert len(response.results) == 1
    assert calls == list(provider.endpoints[:2])


def test_arxiv_network_fallback():
    calls = []

    def factory(endpoint, max_results):
        calls.append(endpoint)
        if len(calls) == 1:
            return FakeClient(
                endpoint,
                error=requests.exceptions.ConnectionError("connection reset"),
            )
        return FakeClient(endpoint, results=[FakeResult()])

    provider = ArxivSearchProvider(
        client_factory=factory,
        search_factory=FakeSearch,
    )
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is True
    assert calls == list(provider.endpoints[:2])


def test_arxiv_all_endpoints_fail_structured():
    calls = []

    def factory(endpoint, max_results):
        calls.append(endpoint)
        return FakeClient(
            endpoint,
            error=arxiv_api.HTTPError(
                f"{endpoint}?q=test",
                0,
                406,
            ),
        )

    provider = ArxivSearchProvider(
        client_factory=factory,
        search_factory=FakeSearch,
    )
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "http"
    assert response.error.status_code == 406
    assert calls == list(provider.endpoints)


def test_arxiv_non_fallback_http_error():
    def factory(endpoint, max_results):
        return FakeClient(
            endpoint,
            error=arxiv_api.HTTPError(
                f"{endpoint}?q=test",
                0,
                400,
            ),
        )

    provider = ArxivSearchProvider(
        client_factory=factory,
        search_factory=FakeSearch,
    )
    provider._last_request_time = 0
    provider.MIN_REQUEST_INTERVAL = 0

    response = provider.search_detailed(SearchQuery(query="transformer"))

    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "http"
    assert response.error.status_code == 400
    assert response.attempts == 1


def test_arxiv_invalid_query():
    provider = ArxivSearchProvider()
    response = provider.search_detailed(SearchQuery(query=""))
    assert response.success is False
    assert response.error is not None
    assert response.error.stage == "validation"


def test_empty_query():
    provider = WikipediaSearchProvider()
    try:
        provider.search(SearchQuery(query=""))
    except ValueError:
        return
    raise AssertionError("空查询应该被拒绝")


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

    provider = WikipediaSearchProvider(http_client=FakeHttpClient())
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
