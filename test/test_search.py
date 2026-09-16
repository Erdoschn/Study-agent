from tools.search import (
    ArxivSearchProvider,
    SearchQuery,
    SearchResult,
    SearchRouter,
)


def test_search_query():
    q = SearchQuery(query="transformer")

    assert q.query == "transformer"
    assert q.source == "arxiv"
    assert q.max_results == 10


def test_arxiv_provider_structure():
    provider = ArxivSearchProvider()

    assert provider.name == "arxiv"


def test_router():
    router = SearchRouter()
    router.register(ArxivSearchProvider())

    assert "arxiv" in router.available_sources()


def test_search_result_structure():
    result = SearchResult(
        source="arxiv",
        title="Test Paper",
        url="https://arxiv.org/abs/1234.5678",
        abstract="Test abstract",
        identifier="1234.5678",
    )

    assert result.source == "arxiv"
    assert result.title == "Test Paper"
    assert result.url.startswith("https://")
    assert result.identifier == "1234.5678"


def test_arxiv_query_building(monkeypatch):
    provider = ArxivSearchProvider()

    captured = {}

    class FakeResponse:
        def read(self):
            return b"""<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <id>http://arxiv.org/abs/1234.5678</id>
                    <title>Test Transformer Paper</title>
                    <summary>Test abstract.</summary>
                    <published>2026-01-01T00:00:00Z</published>
                    <updated>2026-01-02T00:00:00Z</updated>
                    <author><name>Test Author</name></author>
                </entry>
            </feed>"""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "tools.search.arxiv.urllib.request.urlopen",
        fake_urlopen,
    )

    results = provider.search(
        SearchQuery(
            query="transformer",
            categories=["cs.LG"],
            max_results=2,
        )
    )

    assert len(results) == 1
    assert results[0].title == "Test Transformer Paper"
    assert results[0].identifier == "1234.5678"

    assert "transformer" in captured["url"]
    assert "cat%3Acs.LG" in captured["url"]


def test_empty_query():
    provider = ArxivSearchProvider()

    try:
        provider.search(SearchQuery(query=""))
    except ValueError:
        return

    raise AssertionError("空查询应该被拒绝")