from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

from tools.search import ArxivSearchProvider, SearchQuery


def _request_and_print(label: str, url: str, headers: dict[str, str] | None = None) -> None:
    print(f"\n--- {label} ---")
    print(f"URL: {url}")
    request = urllib.request.Request(url=url, method="GET", headers=headers or {})
    print(f"REQUEST HEADERS: {dict(request.header_items())}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(2048).decode("utf-8", errors="replace")
            print(f"HTTP STATUS: {response.status} {response.reason}")
            print(f"RESPONSE HEADERS: {dict(response.headers)}")
            print(f"RESPONSE BODY: {body[:1000]}")
    except urllib.error.HTTPError as exc:
        print(f"HTTP STATUS: {exc.code} {exc.reason}")
        print(f"RESPONSE HEADERS: {dict(exc.headers)}")
        body = exc.read(2048).decode("utf-8", errors="replace")
        print(f"RESPONSE BODY: {body[:1000]}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def diagnose_arxiv_http() -> None:
    base = "https://export.arxiv.org/api/query"
    query = SearchQuery(
        query='"multi-head attention"',
        source="arxiv",
        categories=["cs.LG", "cs.CL"],
        max_results=5,
        sort_by="relevance",
        sort_order="descending",
    )

    search_expression = query.query.strip()
    categories = [c.strip() for c in query.categories if c.strip()]
    if categories:
        category_expression = " OR ".join(f"cat:{category}" for category in categories)
        search_expression = f"({search_expression}) AND ({category_expression})"

    params = {
        "search_query": search_expression,
        "start": 0,
        "max_results": max(1, min(query.max_results, 50)),
        "sortBy": query.sort_by,
        "sortOrder": query.sort_order,
    }
    url = base + "?" + urllib.parse.urlencode(params)

    headers = {
        "User-Agent": (
            "StudyAgent/2.0 (educational research client; "
            "arXiv API search)"
        ),
        "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.1",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

    print("=== 实际 ArxivSearchProvider 查询独立诊断 ===")
    print(f"SEARCH EXPRESSION: {search_expression}")
    print(f"PARAMS: {params}")
    _request_and_print("完整 Provider 请求", url, headers)


def main() -> None:
    diagnose_arxiv_http()
    print("\n=== ArxivSearchProvider 原测试 ===")

    provider = ArxivSearchProvider()
    query = SearchQuery(
        query='"multi-head attention"',
        source="arxiv",
        categories=["cs.LG", "cs.CL"],
        max_results=5,
        sort_by="relevance",
        sort_order="descending",
    )

    try:
        results = provider.search(query)
    except Exception as exc:
        print(f"搜索失败：{exc}")
        return

    print(f"找到 {len(results)} 篇论文：\n")
    for index, item in enumerate(results, 1):
        print(f"[{index}] {item.title}")
        print(f"ID: {item.identifier}")
        print(f"URL: {item.url}")
        print(f"作者: {', '.join(item.authors)}")
        print(f"发布时间: {item.published}")
        print(f"摘要: {item.abstract[:500]}")
        print("-" * 70)


if __name__ == "__main__":
    main()
