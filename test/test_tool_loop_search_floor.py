from core.tool_loop import ToolExecutor


class RecordingSearchRouter:
    def __init__(self):
        self.query = None

    def available_sources(self):
        return ["arxiv"]

    def search(self, query):
        self.query = query
        return []


def test_agent_search_uses_wider_floor_when_reasoner_requests_five():
    router = RecordingSearchRouter()
    executor = ToolExecutor(search_router=router)
    observation = executor.execute(
        "search",
        {"query": "context caching", "source": "arxiv", "max_results": 5},
    )
    assert router.query.max_results == 15
    assert observation.get("results") == []
