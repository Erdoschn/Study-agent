"""Evaluate retrieval on the local BEIR SciFact benchmark.

This benchmark intentionally has no LLM calls. It measures a deterministic
local retriever against SciFact's official qrels so changes to retrieval can
be compared reproducibly.

Usage:
    python scripts/evaluate_scifact.py
    python scripts/evaluate_scifact.py --queries 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "in", "on", "for", "with", "by", "from", "as", "at", "that", "this",
    "these", "those", "it", "its", "their", "than", "into", "using", "used",
}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lower().startswith("query-id"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, relevance = parts[0], parts[1], parts[2]
            try:
                if int(relevance) > 0:
                    qrels[query_id].add(doc_id)
            except ValueError:
                continue
    return dict(qrels)


class BM25:
    """Small dependency-free BM25 implementation for benchmark validation."""

    def __init__(self, documents: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokens(text) for _, text in documents]
        self.avgdl = sum(map(len, self.doc_tokens)) / max(len(self.doc_tokens), 1)
        self.df = Counter()
        for doc in self.doc_tokens:
            self.df.update(set(doc))
        self.n = len(self.doc_tokens)

    def score(self, query: str, index: int) -> float:
        query_terms = set(tokens(query))
        doc = self.doc_tokens[index]
        if not doc:
            return 0.0
        frequencies = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            length_norm = 1 - self.b + self.b * len(doc) / max(self.avgdl, 1e-9)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        return score

    def search(self, query: str, k: int) -> list[str]:
        scored = [(self.score(query, i), doc_id) for i, (doc_id, _) in enumerate(self.documents)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc_id for score, doc_id in scored[:k] if score > 0]


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def select_queries(queries: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0 or count >= len(queries):
        return sorted(queries, key=lambda item: str(item["_id"]))
    rng = random.Random(seed)
    selected = rng.sample(queries, count)
    return sorted(selected, key=lambda item: str(item["_id"]))


def find_qrels(root: Path) -> Path:
    candidates = sorted((root / "qrels").glob("*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No qrels TSV found under {root / 'qrels'}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/scifact"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = args.dataset
    queries = load_jsonl(root / "queries.jsonl")
    corpus = load_jsonl(root / "corpus.jsonl")
    qrels = load_qrels(find_qrels(root))
    selected = select_queries(queries, args.queries, args.seed)

    documents = [
        (
            str(doc["_id"]),
            f"{doc.get('title', '')} {doc.get('text', '')}",
        )
        for doc in corpus
    ]
    retriever = BM25(documents)

    recalls5 = []
    recalls10 = []
    hits5 = []
    hits10 = []
    mrr10 = []
    evaluated = 0
    started = time.perf_counter()

    for query in selected:
        query_id = str(query["_id"])
        relevant = qrels.get(query_id, set())
        if not relevant:
            continue
        ranked = retriever.search(query.get("text", ""), args.top_k)
        recalls5.append(recall_at_k(ranked, relevant, 5))
        recalls10.append(recall_at_k(ranked, relevant, 10))
        hits5.append(hit_at_k(ranked, relevant, 5))
        hits10.append(hit_at_k(ranked, relevant, 10))
        mrr10.append(reciprocal_rank(ranked, relevant, 10))
        evaluated += 1

    elapsed = time.perf_counter() - started

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print("SciFact Retrieval Benchmark")
    print("---------------------------")
    print(f"Queries selected : {len(selected)}")
    print(f"Queries evaluated: {evaluated}")
    print(f"Corpus documents : {len(corpus)}")
    print(f"Seed             : {args.seed}")
    print("Retriever        : dependency-free BM25 baseline")
    print("LLM calls        : 0")
    print(f"Recall@5         : {mean(recalls5):.4f}")
    print(f"Recall@10        : {mean(recalls10):.4f}")
    print(f"Hit@5            : {mean(hits5):.4f}")
    print(f"Hit@10           : {mean(hits10):.4f}")
    print(f"MRR@10           : {mean(mrr10):.4f}")
    print(f"Runtime           : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
