"""secrag ingest | ask | eval | serve"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from secrag.config import Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="secrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Download 10-Ks from EDGAR and build the index")
    ing.add_argument("--no-download", action="store_true", help="Rebuild from data/raw only")

    ask = sub.add_parser("ask", help="Answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--rerank", action="store_true", help="Add cross-encoder reranking (off by default)")

    ev = sub.add_parser("eval", help="Retrieval ablation (+ generation eval with an LLM key)")
    ev.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    ev.add_argument("--out", type=Path, default=Path("eval/results"))
    ev.add_argument("--retrieval-only", action="store_true")
    ev.add_argument("--no-rerank", action="store_true")
    ev.add_argument("--min-recall", type=float, default=None, help="Fail if best-mode recall@k is below this")
    ev.add_argument("--min-faithfulness", type=float, default=None)
    ev.add_argument("--pause", type=float, default=6.0, help="Seconds between questions (free-tier limits)")
    ev.add_argument("--max-questions", type=int, default=None,
                    help="Generation eval on a balanced subset (each question = 2 LLM calls)")

    srv = sub.add_parser("serve", help="Run the API")
    srv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings()

    if args.cmd == "ingest":
        from secrag.pipeline import build_index, default_embedder

        index, report = build_index(s, default_embedder(s), download=not args.no_download)
        for r in report:
            print(f"{r['doc_id']:<20} items={','.join(r['sections'])}  chunks={r['chunks']}")
        print(f"index: {len(index.chunks)} chunks -> {s.index_dir}")
    elif args.cmd == "ask":
        from secrag.pipeline import RAG

        a = RAG.from_settings(s, rerank=args.rerank or None).ask(args.question)
        print(a.text, "\n")
        for src in a.sources():
            print(f"[{src['n']}] {src['ticker']} FY{src['fiscal_year']} Item {src['item']} {src['item_title']}")
    elif args.cmd == "eval":
        return run_eval(args, s)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("secrag.api:app", host="0.0.0.0", port=args.port)
    return 0


def run_eval(args, s: Settings) -> int:
    from secrag.embed import CrossEncoderReranker
    from secrag.evaluate import (
        best_mode,
        generation_eval,
        load_questions,
        retrieval_ablation,
        sample_questions,
        to_markdown,
    )
    from secrag.index import Index
    from secrag.pipeline import default_embedder
    from secrag.retrieve import Retriever

    questions = load_questions(args.questions)
    index = Index.load(s.index_dir)
    reranker = None if args.no_rerank else CrossEncoderReranker(s.reranker_model)
    retriever = Retriever(index, default_embedder(s), reranker, s.candidates)

    ablation = retrieval_ablation(retriever, questions, k=s.top_k)
    gen = None
    if not args.retrieval_only:
        from secrag.generate import GeminiLLM

        llm = GeminiLLM(s.llm_model)
        mode = best_mode(ablation)  # answer with the retrieval mode the ablation measured as best
        gen = generation_eval(retriever, sample_questions(questions, args.max_questions), llm, llm,
                              k=s.top_k, mode=mode, pause_s=args.pause)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((s.index_dir / "meta.json").read_text())
    (args.out / "retrieval.json").write_text(json.dumps(ablation, indent=2))
    if gen:
        (args.out / "generation.json").write_text(json.dumps(gen, indent=2))
    md = to_markdown(ablation, gen, meta)
    (args.out / "results.md").write_text(md)
    print(md)

    failed = []
    best = max(r[f"recall@{s.top_k}"] for r in ablation)
    if args.min_recall is not None and best < args.min_recall:
        failed.append(f"recall@{s.top_k} {best:.3f} < {args.min_recall}")
    judged = sum(1 for r in (gen or {}).get("rows", []) if r.get("faithfulness") is not None)
    if gen and args.min_faithfulness is not None:
        if judged < 5:  # an API quota problem is not a quality regression: report it, don't fail on it
            print(f"WARNING: only {judged} answers judged ({gen.get('stopped_early') or 'see errors'}); "
                  "faithfulness gate skipped")
        elif gen["faithfulness"] < args.min_faithfulness:
            failed.append(f"faithfulness {gen['faithfulness']} < {args.min_faithfulness}")
    if failed:
        print("EVAL GATE FAILED: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
