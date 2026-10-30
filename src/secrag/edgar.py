"""Download 10-K filings from SEC EDGAR (free, no key; a descriptive User-Agent is required)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


@dataclass(frozen=True)
class Filing:
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str  # end of the fiscal period covered
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document)

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}-{self.fiscal_year}-10K"


class EdgarClient:
    """Polite client: SEC allows up to 10 requests/second; we stay well below."""

    def __init__(self, user_agent: str, min_interval_s: float = 0.15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            self._last = time.monotonic()
            resp = self.session.get(url, timeout=60)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    def ticker_map(self) -> dict[str, tuple[int, str]]:
        data = self.get(TICKERS_URL).json()
        return {row["ticker"].upper(): (int(row["cik_str"]), row["title"]) for row in data.values()}

    def latest_10ks(self, ticker: str, cik: int, company: str, n: int) -> list[Filing]:
        recent = self.get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        out = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":  # skip 10-K/A amendments: partial documents
                continue
            out.append(Filing(ticker, company, cik, form, recent["accessionNumber"][i], recent["filingDate"][i],
                              recent["reportDate"][i], recent["primaryDocument"][i]))
            if len(out) == n:
                break
        return out


def download_filings(tickers: list[str], n_per_company: int, raw_dir: Path, user_agent: str) -> list[Filing]:
    """Download (or reuse cached) 10-K HTML + metadata to raw_dir. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = EdgarClient(user_agent)
    tmap = client.ticker_map()
    filings = []
    for ticker in tickers:
        if ticker not in tmap:
            raise ValueError(f"Unknown ticker {ticker}")
        cik, company = tmap[ticker]
        for f in client.latest_10ks(ticker, cik, company, n_per_company):
            html_path, meta_path = raw_dir / f"{f.doc_id}.html", raw_dir / f"{f.doc_id}.json"
            if not html_path.exists():
                log.info("downloading %s %s", f.doc_id, f.url)
                html_path.write_bytes(client.get(f.url).content)
            meta_path.write_text(json.dumps(asdict(f), indent=2))
            filings.append(f)
    return filings


def load_filings(raw_dir: Path) -> list[tuple[Filing, Path]]:
    out = []
    for meta in sorted(raw_dir.glob("*.json")):
        f = Filing(**json.loads(meta.read_text()))
        out.append((f, meta.with_suffix(".html")))
    return out
