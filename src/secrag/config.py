from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

COMPANIES: dict[str, list[str]] = {
    # ticker -> names people use in questions
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon", "aws"],
    "GOOGL": ["alphabet", "google"],
    "META": ["meta", "facebook", "instagram"],
    "TSLA": ["tesla"],
    "NFLX": ["netflix"],
}

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Reserved",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "data/index")))
    tickers: list[str] = field(default_factory=lambda: _env("TICKERS", ",".join(COMPANIES)).split(","))
    filings_per_company: int = field(default_factory=lambda: int(_env("FILINGS_PER_COMPANY", "2")))
    sec_user_agent: str = field(default_factory=lambda: _env(
        "SEC_USER_AGENT", "sec-filings-rag research vishalmankala77@gmail.com"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker_model: str = field(default_factory=lambda: _env(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-3.6-flash"))
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "300")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    candidates: int = field(default_factory=lambda: int(_env("CANDIDATES", "40")))
    # Off by default: on the eval set the cross-encoder lowered hit@1 (0.90 -> 0.81) at ~90x the latency.
    rerank: bool = field(default_factory=lambda: _env("RERANK", "false").lower() in ("1", "true", "yes"))
