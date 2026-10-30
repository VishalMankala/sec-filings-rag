"""FastAPI service.

    POST /ask     {"question": "..."} -> grounded answer + validated citations + sources
    POST /search  {"question": "...", "k": 6, "mode": "hybrid_rerank"} -> retrieved chunks only
    GET  /health
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from secrag.retrieve import MODES

STATE: dict = {}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)


class SearchRequest(AskRequest):
    k: int = Field(6, ge=1, le=20)
    mode: Literal[MODES] = "hybrid_rerank"  # type: ignore[valid-type]


class Source(BaseModel):
    n: int
    chunk_id: str
    ticker: str
    fiscal_year: int
    item: str
    item_title: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations_valid: bool
    uncited_sentences: list[str]
    sources: list[Source]
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "rag" not in STATE:  # tests inject a RAG built from fixtures
        from secrag.config import Settings
        from secrag.pipeline import RAG

        STATE["rag"] = RAG.from_settings(Settings())
    yield


app = FastAPI(title="SEC 10-K RAG", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return resp


@app.get("/health")
def health() -> dict:
    rag = STATE["rag"]
    idx = rag.retriever.index
    return {"status": "ok", "chunks": len(idx.chunks), "documents": len({c.doc_id for c in idx.chunks}),
            "embedder": idx.embedder_name, "llm": rag.llm.name, "mode": rag.mode}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    a = STATE["rag"].ask(req.question)
    return AskResponse(answer=a.text, refused=a.refused, citations_valid=a.citation_valid,
                       uncited_sentences=a.uncited_sentences, sources=[Source(**s) for s in a.sources()],
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.post("/search")
def search(req: SearchRequest) -> list[dict]:
    hits = STATE["rag"].retriever.search(req.question, k=req.k, mode=req.mode)
    return [{"rank": h.rank + 1, "score": round(h.score, 4), "chunk_id": h.chunk.chunk_id,
             "ticker": h.chunk.ticker, "fiscal_year": h.chunk.fiscal_year, "item": h.chunk.item,
             "item_title": h.chunk.item_title, "text": h.chunk.text} for h in hits]
