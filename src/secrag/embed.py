"""Embedding and reranking models behind small interfaces (real models + offline test doubles)."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """BGE-small: 384-dim, strong retrieval quality for its size, runs on CPU."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 500).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.QUERY_PREFIX if "bge" in self.name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True)[0].astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline runs only."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """Reads query and passage together: slower than embeddings, much better at final ordering."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, p) for p in passages], batch_size=32))
