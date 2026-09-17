from tools.search import (
    ArxivSearchProvider,
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


def test_wikipedia_provider_structure():
    provider = WikipediaSearchProvider()

    assert provider.name == "wikipedia"


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
    provider = WikipediaSearchProvider()

    try:
        provider.search(SearchQuery(query=""))
    except ValueError:
        return

    raise AssertionError("空查询应该被拒绝")

from tools.search import SearchStrategy


def test_search_strategy_empty():
    strategy = SearchStrategy()

    decision = strategy.decide("")

    assert decision.need_search is False
    assert decision.confidence == 1.0


def test_search_strategy_realtime():
    strategy = SearchStrategy()

    decision = strategy.decide("现在最新的人工智能新闻是什么？")

    assert decision.need_search is True
    assert decision.source == "wikipedia"
    assert decision.confidence >= 0.9


def test_search_strategy_research():
    strategy = SearchStrategy()

    decision = strategy.decide(
        "Transformer 最近有哪些重要论文？"
    )

    assert decision.need_search is True
    assert decision.source == "arxiv"
    assert decision.confidence >= 0.9


def test_search_strategy_concept():
    strategy = SearchStrategy()

    decision = strategy.decide(
        "什么是 Multi-Head Attention？"
    )

    assert decision.need_search is True
    assert decision.source == "wikipedia"


def test_search_strategy_unknown():
    strategy = SearchStrategy()

    decision = strategy.decide(
        "计算 12345 × 67890"
    )

    assert decision.need_search is False