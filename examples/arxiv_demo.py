from __future__ import annotations

from tools.search import ArxivSearchProvider, SearchQuery


def main() -> None:
    print("=== arXiv provider 实际测试（arxiv.py + requests） ===")

    provider = ArxivSearchProvider()
    query = SearchQuery(
        query='"multi-head attention"',
        source="arxiv",
        categories=["cs.LG", "cs.CL"],
        max_results=5,
        sort_by="relevance",
        sort_order="descending",
    )

    print(f"查询：{query.query}")
    print(f"分类：{query.categories}")
    print("transport：arxiv.py + requests")
    print(f"候选 endpoint：{len(provider.endpoints)}")
    for index, endpoint in enumerate(provider.endpoints, 1):
        print(f"  {index}. {endpoint}")

    response = provider.search_detailed(query)

    print("\n=== SearchResponse ===")
    print(f"success: {response.success}")
    print(f"provider: {response.provider}")
    print(f"attempts: {response.attempts}")
    print(f"elapsed: {response.elapsed_seconds:.2f}s")

    if response.metadata:
        for key in ("transport", "endpoint", "endpoint_index", "endpoint_count"):
            if key in response.metadata:
                print(f"{key}: {response.metadata[key]}")

    if not response.success:
        print("\n=== 搜索失败 ===")
        if response.error:
            print(f"stage: {response.error.stage}")
            print(f"message: {response.error.message}")
            print(f"status_code: {response.error.status_code}")
            print(f"reason: {response.error.reason}")
            print(f"retryable: {response.error.retryable}")
            print(f"attempts: {response.error.attempts}")
        return

    print(f"\n找到 {len(response.results)} 篇论文：\n")
    for index, item in enumerate(response.results, 1):
        print(f"[{index}] {item.title}")
        print(f"ID: {item.identifier}")
        print(f"URL: {item.url}")
        print(f"作者: {', '.join(item.authors)}")
        print(f"发布时间: {item.published}")
        print(f"摘要: {item.abstract[:500]}")
        print("-" * 70)


if __name__ == "__main__":
    main()
