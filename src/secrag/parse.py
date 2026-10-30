"""10-K HTML -> clean text -> sections by Item (1, 1A, 7, 7A, ...).

Real 10-K HTML is messy: inline XBRL tags, hidden metadata, page headers and a table of
contents that repeats every "Item" heading. The section splitter picks, for each Item,
the heading occurrence that starts the longest block of text, which is the real section
and not its line in the table of contents.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup

ITEM_RE = re.compile(
    r"^\s*item\s*(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\s*[\.\:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
PAGE_NOISE_RE = re.compile(r"^(\d{1,3}|page \d+|table of contents)$", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    item: str
    title: str
    text: str


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.lower() in ("ix:header",)):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    # Tables: one line per row, cells separated by " | ", so numbers keep their labels.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n" + "\n".join(rows) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    return normalize(text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or PAGE_NOISE_RE.match(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def split_items(text: str) -> list[Section]:
    lines = text.split("\n")
    heads = []  # (line_index, item, title)
    for i, line in enumerate(lines):
        if len(line) > 200:
            continue
        m = ITEM_RE.match(line)
        if m:
            heads.append((i, m.group(1).upper(), m.group(2).strip(" .")))
    if not heads:
        return [Section("full", "Full document", text)]

    # Length of text from each heading to the next heading of any kind.
    spans = []
    for k, (i, item, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        spans.append((item, title, i, end, len(body)))

    best: dict[str, tuple] = {}
    for span in spans:
        if span[0] not in best or span[4] > best[span[0]][4]:
            best[span[0]] = span

    sections = []
    chosen = sorted(best.values(), key=lambda s: s[2])
    for k, (item, title, start, _, _) in enumerate(chosen):
        end = chosen[k + 1][2] if k + 1 < len(chosen) else len(lines)
        body = "\n".join(lines[start + 1:end]).strip()
        if len(body.split()) < 15:  # "Not applicable." / "None." / reserved
            continue
        if not title and start + 1 < len(lines):
            title = lines[start + 1][:120]
        sections.append(Section(item, title, body))
    return sections


def parse_10k(html: str | bytes) -> list[Section]:
    return split_items(html_to_text(html))
