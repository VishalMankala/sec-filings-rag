"""Glue: build the index from EDGAR, and a ready-to-use RAG object."""

from __future__ import annotations

import logging
from pathlib import Path

from secrag.chunk import chunk_section
from secrag.config import Settings
from secrag.edgar import download_filings, load_filings
from secrag.generate import Answer, answer
from secrag.index import Index
from secrag.parse import parse_10k
from secrag.retrieve import Retriever

log = logging.getLogger(__name__)


def chunks_from_raw(raw_dir: Path, settings: Settings):
    chunks, report = [], []
    for filing, html_path in load_filings(raw_dir):
        sections = parse_10k(html_path.read_bytes())
        n_before = len(chunks)
        for s in sections:
            chunks += chunk_section(s, doc_id=filing.doc_id, ticker=filing.ticker, company=filing.company,
                                    fiscal_year=filing.fiscal_year, url=filing.url,
                                    target_words=settings.chunk_words, overlap_words=settings.chunk_overlap)
        report.append({"doc_id": filing.doc_id, "sections": [s.item for s in sections],
                       "chunks": len(chunks) - n_before})
        log.info("%s: items %s -> %d chunks", filing.doc_id, [s.item for s in sections], len(chunks) - n_before)
    return chunks, report


def build_index(settings: Settings, embedder, download: bool = True) -> tuple[Index, list[dict]]:
    raw = settings.data_dir / "raw"
    if download:
        download_filings(settings.tickers, settings.filings_per_company, raw, settings.sec_user_agent)
    chunks, report = chunks_from_raw(raw, settings)
    if not chunks:
        raise RuntimeError(f"No chunks built from {raw}")
    index = Index.build(chunks, embedder)
    index.save(settings.index_dir)
    return index, report


def default_embedder(settings: Settings):
    from secrag.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.embedding_model)


class RAG:
    def __init__(self, retriever: Retriever, llm, k: int = 6, mode: str = "hybrid"):
        self.retriever, self.llm, self.k, self.mode = retriever, llm, k, mode

    @classmethod
    def from_settings(cls, settings: Settings, llm=None, rerank: bool | None = None) -> RAG:
        from secrag.embed import CrossEncoderReranker
        from secrag.generate import GeminiLLM

        rerank = settings.rerank if rerank is None else rerank
        index = Index.load(settings.index_dir)
        reranker = CrossEncoderReranker(settings.reranker_model) if rerank else None
        retriever = Retriever(index, default_embedder(settings), reranker, settings.candidates)
        return cls(retriever, llm or GeminiLLM(settings.llm_model), settings.top_k,
                   "hybrid_rerank" if rerank else "hybrid")

    def ask(self, question: str) -> Answer:
        hits = self.retriever.search(question, k=self.k, mode=self.mode)
        return answer(question, hits, self.llm)
