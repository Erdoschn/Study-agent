from tools.search import (
    SearchQuery,
    WikipediaSearchProvider,
)


def main():
    provider = WikipediaSearchProvider()

    query = SearchQuery(
        query="Transformer",
        source="wikipedia",
        max_results=3,
    )

    try:
        results = provider.search(query)
    except Exception as exc:
        print(f"搜索失败：{exc}")
        return

    print(f"找到 {len(results)} 个 Wikipedia 条目：\n")

    for index, item in enumerate(results, 1):
        print(f"[{index}] {item.title}")
        print(f"URL: {item.url}")
        print(f"ID: {item.identifier}")
        print(f"摘要: {item.abstract[:500]}")
        print("-" * 70)


if __name__ == "__main__":
    main()