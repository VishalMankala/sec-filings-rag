import numpy as np
import pytest
from conftest import FIX

from secrag.chunk import chunk_section
from secrag.index import Index, tokenize
from secrag.parse import Section, html_to_text, parse_10k, split_items
from secrag.retrieve import QueryFilters, item_boost, parse_filters, rrf


def test_html_to_text_drops_hidden_xbrl_and_keeps_table_rows():
    text = html_to_text((FIX / "acme_10k.html").read_bytes())
    assert "HIDDEN XBRL" not in text and "hidden boilerplate" not in text
    assert "Cloud | 120,500 | 98,000" in text  # "$" cells dropped, label kept with numbers
    assert "\n3\n" not in text  # page numbers removed


def test_split_items_skips_table_of_contents_and_empty_items():
    sections = {s.item: s for s in parse_10k((FIX / "acme_10k.html").read_bytes())}
    assert list(sections) == ["1", "1A", "1C", "7", "7A"]  # 1B "None." dropped
    assert "single-source supplier" in sections["1A"].text
    assert "Chief Information Security Officer" in sections["1C"].text
    assert "Cybersecurity" not in sections["1A"].text  # TOC lines did not leak into sections


def test_split_items_without_headings_returns_full_document():
    assert split_items("just some text\nwith no headings")[0].item == "full"


def test_chunks_respect_sections_size_and_overlap():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = chunk_section(Section("1A", "Risk Factors", body), doc_id="X-2025-10K", ticker="X", company="X Co",
                           fiscal_year=2025, url="u", target_words=100, overlap_words=10)
    assert len(chunks) >= 4
    assert all(c.item == "1A" and c.item_title == "Risk Factors" for c in chunks)
    assert all(len(c.text.split()) <= 100 + 10 + 45 for c in chunks)
    assert chunks[1].text.split()[:10] == chunks[0].text.split()[-10:]  # overlap carried forward
    assert chunks[0].embed_text.startswith("X Co (X) | FY2025 10-K | Item 1A Risk Factors")
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_query_parsing():
    f = parse_filters("What did Acme say about supplier risk in fiscal 2025?")
    assert f.tickers == ["ACME"] and f.years == [2025]
    assert parse_filters("Compare ZETA and acme").tickers == ["ACME", "ZETA"]
    assert parse_filters("general question").tickers == []
    assert "1C" in item_boost("Who oversees cybersecurity?")
    assert "7A" in item_boost("What is the foreign currency exposure?")


def test_rrf_rewards_agreement():
    fused = rrf([np.array([1, 2, 3]), np.array([3, 1, 4])])
    assert max(fused, key=fused.get) == 1  # ranked 1st and 2nd beats 1st and 3rd
    assert set(fused) == {1, 2, 3, 4}


def test_index_save_load_round_trip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert len(loaded.chunks) == len(index.chunks)
    np.testing.assert_allclose(loaded.vectors, index.vectors)
    assert loaded.bm25_scores("supplier").argmax() == index.bm25_scores("supplier").argmax()
    assert "supplier" in tokenize("Single-Source SUPPLIER risk")


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_retrieval_finds_the_right_company_and_section(retriever, mode):
    hits = retriever.search("What does Acme say about its single-source supplier for chips?", k=3, mode=mode)
    assert hits[0].chunk.ticker == "ACME" and hits[0].chunk.item == "1A"


def test_company_filter_is_applied(retriever):
    hits = retriever.search("Who leads the cybersecurity program at Zeta?", k=5, mode="hybrid")
    assert {h.chunk.ticker for h in hits} == {"ZETA"}
    assert hits[0].chunk.item == "1C"


def test_unknown_year_falls_back_to_company(retriever):
    hits = retriever.search("Acme supplier risk in 2019", k=2, mode="hybrid")
    assert hits and hits[0].chunk.ticker == "ACME"


def test_explicit_filters_and_reranker_ordering(index):
    from secrag.embed import HashingEmbedder
    from secrag.retrieve import Retriever

    class ReverseReranker:
        def score(self, query, passages):
            return -np.arange(len(passages), dtype=float)[::-1]  # prefers the last candidate

    r = Retriever(index, HashingEmbedder(), ReverseReranker(), candidates=5)
    plain = r.search("supplier", k=5, mode="hybrid", filters=QueryFilters(["ACME"]))
    reranked = r.search("supplier", k=5, mode="hybrid_rerank", filters=QueryFilters(["ACME"]))
    assert [h.chunk.chunk_id for h in reranked] == [h.chunk.chunk_id for h in plain][::-1]
    with pytest.raises(ValueError):
        r.search("x", mode="magic")
