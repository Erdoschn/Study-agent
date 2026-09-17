from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request

from tools.search import ArxivSearchProvider, SearchQuery


HEADERS = {
    "User-Agent": "StudyAgent/2.0 (educational research client; arXiv API search)",
    "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.1",
    "Accept-Encoding": "identity",
    "Connection": "close",
}


def request_once(label: str, base_url: str) -> None:
    params = {
        "search_query": "all:electron",
        "start": 0,
        "max_results": 1,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"\n--- {label} ---")
    print(f"URL: {url}")
    try:
        request = urllib.request.Request(url=url, method="GET", headers=HEADERS)
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(512).decode("utf-8", errors="replace")
            print(f"HTTP STATUS: {response.status} {response.reason}")
            print(f"CONTENT-TYPE: {response.headers.get('Content-Type')}")
            print(f"RESPONSE BODY: {body[:300]}")
    except urllib.error.HTTPError as exc:
        print(f"HTTP STATUS: {exc.code} {exc.reason}")
        print(f"RESPONSE HEADERS: {dict(exc.headers)}")
        body = exc.read(512).decode("utf-8", errors="replace")
        print(f"RESPONSE BODY: {body[:300]}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def diagnose_arxiv_http() -> None:
    print("=== arXiv 5 秒间隔最小诊断 ===")
    print("只使用 all:electron；每个 endpoint 仅请求 2 次，中间等待 5 秒。")

    for base_url in (
        "https://export.arxiv.org/api/query",
        "https://arxiv.org/api/query",
    ):
        host = urllib.parse.urlparse(base_url).netloc
        request_once(f"{host} · 第 1 次", base_url)
        print("等待 5 秒，不发送任何请求……")
        time.sleep(5)
        request_once(f"{host} · 5 秒后第 2 次", base_url)


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
