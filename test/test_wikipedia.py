from tools.search.http import HttpResponse
from tools.search.models import SearchQuery
from tools.search.wikipedia import WikipediaSearchProvider


class FakeHttp:
    def __init__(self):
        self.calls = 0

    def get(self, url, headers=None):
        self.calls += 1
        if self.calls == 1:
            body = b"""{"query":{"search":[{"title":"Attention"}]}}"""
            return HttpResponse(200, "OK", {}, body, attempts=2, elapsed_seconds=0.1)
        body = b"""{"query":{"pages":{"1":{"pageid":1,"title":"Attention","extract":"Attention mechanisms"}}}}"""
        return HttpResponse(200, "OK", {}, body, attempts=3, elapsed_seconds=0.1)


def test_wikipedia_search_accumulates_attempts_across_requests():
    provider = WikipediaSearchProvider(http_client=FakeHttp())
    response = provider.search_detailed(
        SearchQuery(query="attention", source="wikipedia")
    )

    assert response.success is True
    assert response.attempts == 5
    assert len(response.results) == 1
