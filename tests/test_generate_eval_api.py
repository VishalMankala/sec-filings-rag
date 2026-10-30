import json

from conftest import FakeLLM
from fastapi.testclient import TestClient

from secrag.evaluate import EvalQuestion, generation_eval, judge, load_questions, retrieval_ablation, to_markdown
from secrag.generate import REFUSAL, SYSTEM_PROMPT, answer, build_prompt, check_citations
from secrag.pipeline import RAG

Q = "What does Acme say about its single-source supplier?"


def test_prompt_numbers_sources_and_states_rules(retriever):
    hits = retriever.search(Q, k=3, mode="hybrid")
    prompt = build_prompt(Q, hits)
    assert "[1] Acme Robotics Inc. (ACME) | FY2025 10-K | Item 1A" in prompt
    assert "[3]" in prompt and "[4]" not in prompt
    assert REFUSAL in SYSTEM_PROMPT and "ONLY" in SYSTEM_PROMPT


def test_check_citations():
    valid, invalid, uncited = check_citations(
        "Acme relies on one supplier [1]. Revenue grew 23% [2][3]. It also has a factory in Ohio near the lake. "
        "See [9].", n_sources=3)
    assert valid == [1, 2, 3] and invalid == [9]
    assert uncited == ["It also has a factory in Ohio near the lake."]


def test_answer_with_fake_llm(retriever):
    a = answer(Q, retriever.search(Q, k=3, mode="hybrid"), FakeLLM())
    assert not a.refused and a.citation_valid and a.cited == [1]
    assert a.sources()[0]["ticker"] == "ACME" and a.sources()[0]["item"] == "1A"


def test_refusal_and_empty_retrieval(retriever):
    a = answer("What is Initech's revenue?", retriever.search(Q, k=2, mode="hybrid"), FakeLLM(REFUSAL))
    assert a.refused and a.uncited_sentences == []
    llm = FakeLLM()
    empty = answer("anything", [], llm)
    assert empty.refused and llm.calls == []  # no LLM call when nothing was retrieved


def test_judge_parses_json_and_tolerates_garbage():
    ok = FakeLLM('Sure! {"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}], '
                 '"relevance": 4}')
    r = judge(ok, "q", "ctx", "ans")
    assert (r["claims"], r["supported"], r["faithfulness"], r["relevance"]) == (2, 1, 0.5, 4)
    assert judge(FakeLLM("not json"), "q", "c", "a")["faithfulness"] is None


def test_retrieval_ablation_and_generation_eval(retriever, tmp_path):
    questions = [
        EvalQuestion("q1", Q, ["ACME"], ["1A"]),
        EvalQuestion("q2", "How does Zeta hedge variable rate debt?", ["ZETA"], ["7A"]),
        EvalQuestion("q3", "Compare Acme and Zeta cybersecurity leadership", ["ACME", "ZETA"], ["1C"]),
        EvalQuestion("u1", "What is Initech's revenue?", [], [], answerable=False),
    ]
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(q.__dict__) for q in questions))
    assert len(load_questions(p)) == 4

    ablation = retrieval_ablation(retriever, questions, k=3)
    assert [r["mode"] for r in ablation] == ["bm25", "dense", "hybrid"]  # no reranker -> skipped
    hybrid = ablation[-1]
    assert hybrid["questions"] == 3 and hybrid["recall@3"] == 1.0 and hybrid["hit@1"] >= 2 / 3

    judge_llm = FakeLLM('{"claims": [{"claim": "x", "supported": true}], "relevance": 5}')
    gen = generation_eval(retriever, questions, FakeLLM(), judge_llm, k=3)
    assert gen["faithfulness"] == 1.0 and gen["citation_validity"] == 1.0
    assert gen["correct_refusal_rate"] == 0.0  # the fake LLM never refuses: the metric catches it
    md = to_markdown(ablation, gen, {"n_chunks": 10, "documents": ["a", "b"], "embedder": "hashing"})
    assert "Retrieval ablation" in md and "Faithfulness" in md


def test_api(retriever):
    from secrag import api

    api.STATE["rag"] = RAG(retriever, FakeLLM(), k=3, mode="hybrid")
    with TestClient(api.app) as client:
        assert client.get("/health").json()["chunks"] == len(retriever.index.chunks)
        r = client.post("/ask", json={"question": Q})
        body = r.json()
        assert r.status_code == 200 and not body["refused"] and body["citations_valid"]
        assert body["sources"][0]["ticker"] == "ACME"
        assert "X-Process-Time-Ms" in r.headers
        hits = client.post("/search", json={"question": Q, "k": 2, "mode": "bm25"}).json()
        assert len(hits) == 2 and hits[0]["item"] == "1A"
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        assert client.post("/search", json={"question": Q, "mode": "magic"}).status_code == 422
    api.STATE.clear()


class FlakyError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def test_with_retries_recovers_from_overload_and_gives_up_on_real_errors():
    from secrag.generate import with_retries

    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise FlakyError(503)
        return "ok"

    assert with_retries(flaky, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]  # exponential backoff

    def bad_request():
        raise FlakyError(400)

    import pytest

    with pytest.raises(FlakyError):
        with_retries(bad_request, sleep=sleeps.append)

    def always_down():
        raise FlakyError(503)

    with pytest.raises(FlakyError):
        with_retries(always_down, attempts=3, sleep=lambda s: None)


def test_generation_eval_skips_questions_whose_llm_calls_fail(retriever):
    class HalfBrokenLLM(FakeLLM):
        def complete(self, system, prompt):
            if "Zeta" in prompt.split("Question:")[-1]:
                raise FlakyError(503)
            return super().complete(system, prompt)

    qs = [EvalQuestion("ok", Q, ["ACME"], ["1A"]),
          EvalQuestion("down", "How does Zeta hedge variable rate debt?", ["ZETA"], ["7A"])]
    judge_llm = FakeLLM('{"claims": [{"claim": "x", "supported": true}], "relevance": 5}')
    gen = generation_eval(retriever, qs, HalfBrokenLLM(), judge_llm, k=3, mode="hybrid")
    assert gen["questions"] == 2 and gen["llm_errors"] == 1 and gen["errors"][0]["id"] == "down"
    assert gen["faithfulness"] == 1.0


def test_best_mode_prefers_mrr_then_speed():
    from secrag.evaluate import best_mode

    ablation = [{"mode": "bm25", "mrr": 0.837, "ms_per_query": 3.7},
                {"mode": "hybrid", "mrr": 0.936, "ms_per_query": 34.1},
                {"mode": "hybrid_rerank", "mrr": 0.891, "ms_per_query": 2980.5}]
    assert best_mode(ablation) == "hybrid"
    assert best_mode([{"mode": "a", "mrr": 0.9, "ms_per_query": 50},
                      {"mode": "b", "mrr": 0.9, "ms_per_query": 5}]) == "b"


def test_generation_eval_stops_early_when_quota_is_exhausted(retriever):
    class QuotaLLM(FakeLLM):
        def complete(self, system, prompt):
            raise FlakyError(429)

    qs = [EvalQuestion(f"q{i}", Q, ["ACME"], ["1A"]) for i in range(10)]
    gen = generation_eval(retriever, qs, QuotaLLM(), QuotaLLM(), k=3, mode="hybrid")
    assert gen["evaluated"] == 0 and gen["llm_errors"] == 2
    assert "not attempted" in gen["stopped_early"] and gen["faithfulness"] is None


def test_sample_questions_keeps_unanswerable_and_spreads_the_rest():
    from secrag.evaluate import sample_questions

    qs = [EvalQuestion(f"a{i}", "q", ["X"], ["1"]) for i in range(40)]
    qs += [EvalQuestion(f"u{i}", "q", [], [], answerable=False) for i in range(6)]
    picked = sample_questions(qs, 20)
    assert len(picked) == 20
    assert sum(not q.answerable for q in picked) == 6
    assert picked[0].id == "a0" and picked[13].id not in {"a13"}  # spread, not the first 14
    assert sample_questions(qs, None) == qs
