"""Hybrid index: BM25 (exact terms, tickers, numbers) + dense vectors (meaning)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from secrag.chunk import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "our we us they their which what how does did do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOPWORDS]


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder_name = embedder_name
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks]) if chunks else None
        self.by_id = {c.chunk_id: i for i, c in enumerate(chunks)}

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> Index:
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        return cls(chunks, vectors, embedder.name)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "chunks.jsonl", "w") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c)) + "\n")
        (path / "meta.json").write_text(json.dumps({
            "embedder": self.embedder_name, "n_chunks": len(self.chunks),
            "documents": sorted({c.doc_id for c in self.chunks}),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> Index:
        meta = json.loads((path / "meta.json").read_text())
        with open(path / "chunks.jsonl") as fh:
            chunks = [Chunk(**json.loads(line)) for line in fh]
        return cls(chunks, np.load(path / "vectors.npy"), meta["embedder"])

    # ---- raw scoring -----------------------------------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.vectors @ query_vec  # vectors are L2-normalized: dot product = cosine
