"""Query understanding + hybrid retrieval + reciprocal rank fusion + optional reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from secrag.chunk import Chunk
from secrag.config import COMPANIES
from secrag.index import Index

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
ITEM_HINTS = {
    "1A": ["risk", "risks", "risk factor", "threat", "uncertain"],
    "1C": ["cybersecurity", "cyber", "information security", "ciso"],
    "7": ["revenue", "results of operations", "margin", "liquidity", "capital expenditure", "md&a", "outlook"],
    "7A": ["interest rate risk", "foreign currency", "foreign exchange", "market risk", "hedg"],
    "3": ["legal proceeding", "lawsuit", "litigation"],
    "5": ["repurchase", "buyback", "dividend", "share repurchase"],
    "1": ["business", "segment", "products", "employees", "competition", "human capital"],
}


@dataclass
class QueryFilters:
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)

    def matches(self, c: Chunk) -> bool:
        return (not self.tickers or c.ticker in self.tickers) and (not self.years or c.fiscal_year in self.years)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def parse_filters(question: str) -> QueryFilters:
    """Pull company and fiscal-year constraints out of the question."""
    q = question.lower()
    tickers = []
    for ticker, names in COMPANIES.items():
        if re.search(rf"\b{ticker.lower()}\b", q) or any(re.search(rf"\b{re.escape(n)}\b", q) for n in names):
            tickers.append(ticker)
    years = sorted({int(y) for y in re.findall(r"\b(?:fy\s?|fiscal\s(?:year\s)?)?(20[1-3]\d)\b", q)})
    return QueryFilters(tickers, years)


def item_boost(question: str) -> set[str]:
    q = question.lower()
    return {item for item, words in ITEM_HINTS.items() if any(w in q for w in words)}


def rrf(rankings: list[np.ndarray], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: robust to the very different score scales of BM25 and cosine."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self, index: Index, embedder, reranker=None, candidates: int = 40):
        self.index, self.embedder, self.reranker, self.candidates = index, embedder, reranker, candidates

    def search(self, question: str, k: int = 6, mode: str = "hybrid_rerank",
               filters: QueryFilters | None = None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        filters = filters if filters is not None else parse_filters(question)
        allowed = np.array([filters.matches(c) for c in self.index.chunks])
        if not allowed.any():  # e.g. a year we have not indexed: fall back to company only
            filters = QueryFilters(filters.tickers, [])
            allowed = np.array([filters.matches(c) for c in self.index.chunks])
        if not allowed.any():
            return []

        def ranked(scores: np.ndarray) -> np.ndarray:
            scores = np.where(allowed, scores, -np.inf)
            order = np.argsort(-scores, kind="stable")
            return order[: min(self.candidates, int(allowed.sum()))]

        bm25_rank = ranked(self.index.bm25_scores(question)) if mode != "dense" else None
        dense_rank = ranked(self.index.dense_scores(self.embedder.embed_query(question))) if mode != "bm25" else None

        if mode == "bm25":
            fused = {int(i): 1.0 / (r + 1) for r, i in enumerate(bm25_rank)}
        elif mode == "dense":
            fused = {int(i): 1.0 / (r + 1) for r, i in enumerate(dense_rank)}
        else:
            fused = rrf([bm25_rank, dense_rank])
            boost = item_boost(question)
            if boost:  # gentle prior toward the section that usually answers this kind of question
                fused = {i: s * (1.15 if self.index.chunks[i].item in boost else 1.0) for i, s in fused.items()}

        cand = sorted(fused, key=lambda i: -fused[i])[: self.candidates]
        if mode == "hybrid_rerank" and self.reranker is not None and cand:
            scores = self.reranker.score(question, [self.index.chunks[i].embed_text for i in cand])
            order = np.argsort(-scores, kind="stable")
            cand, final = [cand[j] for j in order], [float(scores[j]) for j in order]
        else:
            final = [fused[i] for i in cand]
        return [Hit(self.index.chunks[i], s, r) for r, (i, s) in enumerate(zip(cand[:k], final[:k], strict=False))]
