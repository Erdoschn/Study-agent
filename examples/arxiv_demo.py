from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tools.search import (
    ArxivSearchProvider,
    SearchQuery,
)


def diagnose_arxiv_http():
    urls = [
        "http://export.arxiv.org/api/query?search_query=all%3Aelectron&start=0&max_results=1",
        "https://export.arxiv.org/api/query?search_query=all%3Aelectron&start=0&max_results=1",
    ]

    print("=== HTTP / HTTPS 对照诊断 ===")

    for url in urls:
        print(f"\nURL: {url}")
        request = Request(url, method="GET")
        request.add_header("User-Agent", "StudyAgent/2.0")

        try:
            with urlopen(request, timeout=30) as response:
                body = response.read(2048).decode("utf-8", errors="replace")
                print(f"HTTP STATUS: {response.status} {response.reason}")
                print(f"RESPONSE HEADERS: {dict(response.headers)}")
                print(f"RESPONSE BODY: {body[:1000]}")
        except HTTPError as exc:
            print(f"HTTP STATUS: {exc.code} {exc.reason}")
            print(f"RESPONSE HEADERS: {dict(exc.headers)}")
            body = exc.read(2048).decode("utf-8", errors="replace")
            print(f"RESPONSE BODY: {body[:1000]}")
        except URLError as exc:
            print(f"URL ERROR: {exc}")
        except Exception as exc:
            print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")


def main():
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
