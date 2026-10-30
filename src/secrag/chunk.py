"""Section-aware chunking with contextual headers.

Chunks never cross an Item boundary, and each one carries a header such as
"Apple Inc. | FY2025 10-K | Item 1A Risk Factors". A paragraph about "supply
constraints" in Apple's risk factors then embeds near questions about Apple's risks,
not near every company's generic supply-chain text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from secrag.config import ITEM_TITLES
from secrag.parse import Section


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    item_title: str
    text: str  # body only (shown to users, cited)
    url: str

    @property
    def header(self) -> str:
        return f"{self.company} ({self.ticker}) | FY{self.fiscal_year} 10-K | Item {self.item} {self.item_title}"

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z•●])", text) if p.strip()]
    out = []
    for p in paras:  # hard-split very long paragraphs on sentence boundaries
        words = p.split()
        if len(words) <= 400:
            out.append(p)
            continue
        sents, cur = re.split(r"(?<=[.!?])\s+", p), []
        for s in sents:
            cur.append(s)
            if sum(len(x.split()) for x in cur) > 250:
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
    return out


def chunk_section(section: Section, *, doc_id: str, ticker: str, company: str, fiscal_year: int, url: str,
                  target_words: int = 300, overlap_words: int = 50) -> list[Chunk]:
    chunks, buf, buf_words = [], [], 0
    title = ITEM_TITLES.get(section.item, section.title)

    def emit():
        text = "\n".join(buf).strip()
        if len(text.split()) < 20:
            return
        cid = hashlib.sha1(f"{doc_id}|{section.item}|{len(chunks)}|{text[:80]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(cid, doc_id, ticker, company, fiscal_year, section.item, title, text, url))

    for para in _paragraphs(section.text):
        n = len(para.split())
        if buf and buf_words + n > target_words:
            emit()
            tail = " ".join(" ".join(buf).split()[-overlap_words:]) if overlap_words else ""
            buf, buf_words = ([tail] if tail else []), len(tail.split())
        buf.append(para)
        buf_words += n
    if buf:
        emit()
    return chunks
