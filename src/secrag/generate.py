"""Grounded answer generation with numbered citations, plus citation validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from secrag.retrieve import Hit

REFUSAL = "I can't find this in the indexed 10-K filings."

SYSTEM_PROMPT = f"""You are a careful equity research assistant answering questions about companies' \
annual reports (Form 10-K).

Rules:
1. Use ONLY the numbered excerpts provided. Do not use outside knowledge, even if you are confident.
2. Put a citation like [1] or [2][4] after every sentence that states a fact from the excerpts.
3. Quote figures exactly as written (units, periods). Never compute numbers that are not in the excerpts \
unless the question asks for it, and then show the inputs with their citations.
4. If the excerpts do not contain the answer, reply exactly: "{REFUSAL}"
5. If excerpts cover only part of the question, answer that part and say what is missing.
6. Be concise: at most about 150 words unless the question asks for detail."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class GeminiLLM:
    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.0):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)")
        self.client = genai.Client(api_key=key)
        self.name, self.temperature = model, temperature

    def complete(self, system: str, prompt: str) -> str:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.name, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature),
        )
        return (resp.text or "").strip()


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        blocks.append(f"[{n}] {h.chunk.header}\n{h.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return f"Excerpts:\n\n{format_context(hits)}\n\n---\n\nQuestion: {question}\nAnswer:"


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    refused: bool = False

    @property
    def citation_valid(self) -> bool:
        return not self.invalid_citations

    def sources(self) -> list[dict]:
        return [{"n": n, "chunk_id": self.hits[n - 1].chunk.chunk_id, "ticker": self.hits[n - 1].chunk.ticker,
                 "fiscal_year": self.hits[n - 1].chunk.fiscal_year, "item": self.hits[n - 1].chunk.item,
                 "item_title": self.hits[n - 1].chunk.item_title, "url": self.hits[n - 1].chunk.url,
                 "snippet": self.hits[n - 1].chunk.text[:400]}
                for n in self.cited if 1 <= n <= len(self.hits)]


CITE_RE = re.compile(r"\[(\d{1,2})\]")


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int], list[str]]:
    """Return (valid cited numbers, out-of-range numbers, factual-looking sentences with no citation)."""
    nums = [int(x) for x in CITE_RE.findall(text)]
    valid = sorted({n for n in nums if 1 <= n <= n_sources})
    invalid = sorted({n for n in nums if not 1 <= n <= n_sources})
    uncited = []
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        s = sent.strip()
        if len(s.split()) >= 6 and not CITE_RE.search(s) and REFUSAL not in s and not s.endswith(":"):
            uncited.append(s)
    return valid, invalid, uncited


def answer(question: str, hits: list[Hit], llm: LLM) -> Answer:
    if not hits:
        return Answer(question, REFUSAL, [], refused=True)
    text = llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))
    refused = REFUSAL.lower().rstrip(".") in text.lower()
    valid, invalid, uncited = check_citations(text, len(hits))
    return Answer(question, text, hits, valid, invalid, [] if refused else uncited, refused)
