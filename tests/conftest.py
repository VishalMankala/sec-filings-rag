from pathlib import Path

import pytest

from secrag.chunk import chunk_section
from secrag.embed import HashingEmbedder
from secrag.index import Index
from secrag.parse import parse_10k
from secrag.retrieve import Retriever

FIX = Path(__file__).parent / "fixtures"
DOCS = {  # ticker -> (company, fiscal year, fixture)
    "ACME": ("Acme Robotics Inc.", 2025, "acme_10k.html"),
    "ZETA": ("Zeta Foods Corp.", 2024, "zeta_10k.html"),
}


@pytest.fixture(autouse=True)
def fixture_companies(monkeypatch):
    """Make the query parser recognise the fixture companies."""
    import secrag.retrieve as r

    monkeypatch.setattr(r, "COMPANIES", {"ACME": ["acme"], "ZETA": ["zeta"]})


@pytest.fixture(scope="session")
def chunks():
    out = []
    for ticker, (company, fy, fname) in DOCS.items():
        for s in parse_10k((FIX / fname).read_bytes()):
            out += chunk_section(s, doc_id=f"{ticker}-{fy}-10K", ticker=ticker, company=company, fiscal_year=fy,
                                 url=f"https://example.com/{ticker}", target_words=60, overlap_words=10)
    return out


@pytest.fixture(scope="session")
def index(chunks):
    return Index.build(chunks, HashingEmbedder())


@pytest.fixture
def retriever(index):
    return Retriever(index, HashingEmbedder(), reranker=None, candidates=20)


class FakeLLM:
    """Echoes the first sentence of the first excerpt with a citation, or refuses when asked."""

    name = "fake-llm"

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if self.reply is not None:
            return self.reply
        first = prompt.split("[1] ", 1)[1].split("\n", 2)[1]
        return first.split(". ")[0].rstrip(".") + ". [1]"
