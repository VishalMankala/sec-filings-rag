"""Evaluation harness.

Retrieval (no API key needed): for every question we know which company and which 10-K
Item should contain the answer. A retrieved chunk is relevant if it matches both.
    recall@k   share of the question's companies with a relevant chunk in the top k
    hit@1      the first result is relevant
    MRR        1 / rank of the first relevant chunk

Generation (needs an LLM): citation validity and coverage are checked deterministically;
faithfulness and relevance are judged by an LLM that sees ONLY the retrieved excerpts;
unanswerable questions must be refused.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from secrag.generate import LLM, answer, format_context
from secrag.retrieve import MODES, Retriever


@dataclass
class EvalQuestion:
    id: str
    question: str
    tickers: list[str]
    items: list[str]
    answerable: bool = True
    category: str = "general"


def load_questions(path: Path) -> list[EvalQuestion]:
    with open(path) as fh:
        return [EvalQuestion(**json.loads(line)) for line in fh if line.strip()]


def _relevant(chunk, q: EvalQuestion) -> bool:
    return chunk.ticker in q.tickers and chunk.item in q.items


def retrieval_metrics(retriever: Retriever, questions: list[EvalQuestion], mode: str, k: int = 6) -> dict:
    rows = []
    t0 = time.perf_counter()
    for q in (q for q in questions if q.answerable):
        hits = retriever.search(q.question, k=k, mode=mode)
        covered = {h.chunk.ticker for h in hits if _relevant(h.chunk, q)}
        first = next((h.rank + 1 for h in hits if _relevant(h.chunk, q)), None)
        rows.append({"id": q.id, "recall": len(covered) / len(q.tickers), "hit1": bool(hits and _relevant(
            hits[0].chunk, q)), "rr": 1 / first if first else 0.0,
            "top": [f"{h.chunk.ticker}/{h.chunk.fiscal_year}/{h.chunk.item}" for h in hits[:3]]})
    n = len(rows)
    return {
        "mode": mode, "k": k, "questions": n,
        f"recall@{k}": round(sum(r["recall"] for r in rows) / n, 4),
        "hit@1": round(sum(r["hit1"] for r in rows) / n, 4),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
        "ms_per_query": round((time.perf_counter() - t0) * 1000 / max(n, 1), 1),
        "misses": [r for r in rows if r["recall"] < 1],
    }


def best_mode(ablation: list[dict]) -> str:
    """Highest MRR wins; on a tie, the cheaper (faster) mode."""
    return max(ablation, key=lambda r: (round(r["mrr"], 3), -r["ms_per_query"]))["mode"]


def retrieval_ablation(retriever: Retriever, questions: list[EvalQuestion], k: int = 6) -> list[dict]:
    modes = [m for m in MODES if m != "hybrid_rerank" or retriever.reranker is not None]
    return [retrieval_metrics(retriever, questions, m, k) for m in modes]


JUDGE_SYSTEM = """You grade answers produced by a retrieval-augmented system. You see the question, \
the excerpts the system was given, and its answer. Judge ONLY against the excerpts, not world knowledge.
Return JSON only:
{"claims": [{"claim": "<one factual claim from the answer>", "supported": true|false}],
 "relevance": <1-5: does the answer address the question asked?>}
A claim is supported only if the excerpts state it (paraphrase is fine; invented numbers are not).
List every factual claim in the answer. A refusal has no claims."""


def judge(llm: LLM, question: str, context: str, answer_text: str) -> dict:
    raw = llm.complete(JUDGE_SYSTEM, f"Question: {question}\n\nExcerpts:\n{context}\n\nAnswer:\n{answer_text}")
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    claims = data.get("claims") or []
    supported = sum(1 for c in claims if c.get("supported") is True)
    return {"claims": len(claims), "supported": supported,
            "faithfulness": supported / len(claims) if claims else None,
            "relevance": data.get("relevance")}


def generation_eval(retriever: Retriever, questions: list[EvalQuestion], llm: LLM, judge_llm: LLM,
                    k: int = 6, mode: str = "hybrid_rerank", pause_s: float = 0.0) -> dict:
    if mode == "hybrid_rerank" and retriever.reranker is None:
        mode = "hybrid"
    rows, errors = [], []
    for q in questions:
        hits = retriever.search(q.question, k=k, mode=mode)
        try:
            t0 = time.perf_counter()
            a = answer(q.question, hits, llm)
            latency = time.perf_counter() - t0
            row = {"id": q.id, "answerable": q.answerable, "refused": a.refused,
                   "citation_valid": a.citation_valid, "cited": len(a.cited),
                   "uncited_sentences": len(a.uncited_sentences), "latency_s": round(latency, 2),
                   "answer": a.text}
            if q.answerable and not a.refused:
                row.update(judge(judge_llm, q.question, format_context(hits), a.text))
            rows.append(row)
        except Exception as err:  # noqa: BLE001 - one API outage must not sink the whole evaluation
            errors.append({"id": q.id, "error": f"{type(err).__name__}: {err}"[:300]})
        if pause_s:
            time.sleep(pause_s)  # stay under free-tier rate limits
    if not rows:
        raise RuntimeError(f"Every LLM call failed; first error: {errors[0]['error'] if errors else '?'}")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    faith = [r["faithfulness"] for r in ans if r.get("faithfulness") is not None]
    rel = [r["relevance"] for r in ans if isinstance(r.get("relevance"), int | float)]
    return {
        "mode": mode, "llm": llm.name, "judge": judge_llm.name, "questions": len(rows) + len(errors),
        "faithfulness": round(statistics.mean(faith), 4) if faith else None,
        "answer_relevance_1to5": round(statistics.mean(rel), 2) if rel else None,
        "answered_rate": round(sum(not r["refused"] for r in ans) / len(ans), 4) if ans else None,
        "correct_refusal_rate": round(sum(r["refused"] for r in unans) / len(unans), 4) if unans else None,
        "citation_validity": round(sum(r["citation_valid"] for r in rows) / len(rows), 4),
        "answers_with_uncited_sentences": round(
            sum(r["uncited_sentences"] > 0 for r in ans if not r["refused"]) / max(1, sum(not r["refused"]
                                                                                          for r in ans)), 4),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "llm_errors": len(errors),
        "errors": errors,
        "rows": rows,
    }


def to_markdown(ablation: list[dict], gen: dict | None, index_meta: dict) -> str:
    k = ablation[0]["k"]
    lines = [
        "# Evaluation results", "",
        f"Index: {index_meta.get('n_chunks', '?')} chunks from {len(index_meta.get('documents', []))} 10-K filings; "
        f"embedder `{index_meta.get('embedder')}`.", "",
        "## Retrieval ablation", "",
        f"| Mode | Recall@{k} | Hit@1 | MRR | ms/query |", "|---|---:|---:|---:|---:|",
    ]
    for r in ablation:
        lines.append(f"| {r['mode']} | {r[f'recall@{k}']:.3f} | {r['hit@1']:.3f} | {r['mrr']:.3f} | "
                     f"{r['ms_per_query']} |")
    if gen:
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

        lines += ["", f"## Generation ({gen['llm']}, judged by {gen['judge']}, retrieval `{gen['mode']}`)", "",
                  "| Metric | Value |", "|---|---:|",
                  f"| Faithfulness (claims supported by excerpts) | {fmt(gen['faithfulness'])} |",
                  f"| Answer relevance (1-5) | {fmt(gen['answer_relevance_1to5'])} |",
                  f"| Answered rate (answerable questions) | {fmt(gen['answered_rate'])} |",
                  f"| Correct refusal rate (unanswerable questions) | {fmt(gen['correct_refusal_rate'])} |",
                  f"| Citation validity | {fmt(gen['citation_validity'])} |",
                  f"| Answers with an uncited sentence | {fmt(gen['answers_with_uncited_sentences'])} |",
                  f"| Median generation latency (s) | {fmt(gen['p50_latency_s'])} |",
                  f"| Questions skipped (LLM API errors after retries) | {gen['llm_errors']} |"]
    return "\n".join(lines) + "\n"
