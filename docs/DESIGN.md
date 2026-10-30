# Design notes

## Why 10-Ks

They are long (often 50k+ words), structured by law into numbered Items, mix prose with tables, and
reward precise citation. The structure is free supervision: for many questions we know *which Item*
must contain the answer (risks → 1A, cyber governance → 1C, segment results → 7, FX/interest-rate
exposure → 7A). That makes retrieval measurable without hand-labelling passages.

## Parsing

| Problem | Handling |
|---|---|
| Inline XBRL (`ix:header`) and `display:none` blocks | Removed before text extraction |
| Tables lose meaning as plain text | One line per row, `label \| v1 \| v2`; lone `$`/`%` cells dropped |
| Table of contents repeats every Item heading | For each Item, keep the occurrence that starts the longest section |
| Page numbers / "Table of Contents" running headers | Dropped line by line |
| Items with "None." / "Not applicable." | Dropped (fewer than 15 words) |

Limitation: a few filers incorporate MD&A or financial statements *by reference* from an annual
report exhibit. Those sections are thin in the primary document. The per-filing ingest report lists
which Items were found, so this is visible.

## Chunking

About 300 words with a 50-word overlap, split on paragraph boundaries, never across Items. Each chunk
is embedded with a contextual header (company, ticker, fiscal year, Item). Without it, risk-factor
boilerplate from different companies is almost indistinguishable to an embedding model.

## Retrieval

- **BM25** (Okapi) over header + text: exact matches for tickers, product names, "Reality Labs", numbers.
- **Dense**: `BAAI/bge-small-en-v1.5` (384-d, CPU-friendly) with its query instruction prefix,
  normalized vectors, exact cosine via one matrix product. At a few thousand chunks, an ANN index (FAISS/HNSW)
  would add complexity without a measurable gain. It becomes worth it beyond about 1M chunks.
- **Reciprocal rank fusion** (k = 60) instead of a weighted score sum: BM25 scores are unbounded and
  depend on the query, while cosine lies in [-1, 1]. RRF needs only ranks, so there is no weight to tune.
- **Metadata filters** from the question (company names, tickers, fiscal years). If a year filter
  matches nothing, it falls back to company-only instead of returning nothing.
- **Section prior**: a mild ×1.15 boost when the question's wording suggests an Item (e.g. "cybersecurity"
  → 1C). It is deliberately small so retrieval quality, not the heuristic, decides.
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) of the top 40 fused candidates. It reads
  question and passage together, which is more accurate than embeddings but too slow to run over every
  chunk. That's why it runs only on the fused candidates.

The ablation in the evaluation report measures what each step contributes, instead of assuming it helps.
It paid off: the generic MS MARCO reranker *lowered* hit@1 from 0.90 to 0.81 on 10-K questions at ~90x the
latency. MS MARCO teaches "does this web passage answer this web query"; 10-K sections are long, formal and
full of near-duplicate boilerplate across companies, where the reranker over-weights generic risk language.
So serving uses plain hybrid retrieval, and the evaluation chooses the mode for generation from the measured
MRR rather than from a hard-coded default.

**Resilience.** Free-tier LLM APIs return 429/503 under load. Every Gemini call retries with exponential
backoff and jitter (5 s, 10 s, 20 s, …), and a question whose calls still fail is recorded as skipped instead
of aborting the whole evaluation. The report states how many were skipped.

## Generation

- Temperature 0, numbered excerpts, explicit rules: cite every claim, quote numbers exactly, refuse
  with a fixed sentence when the answer is missing, and say which part is missing for partial answers.
- If retrieval returns nothing (e.g. an unknown company), no LLM call is made.
- **Post-hoc citation checks** in code: out-of-range citation numbers and factual sentences without
  any citation are flagged in the response rather than silently trusted.

## Evaluation

| Metric | How | Needs LLM |
|---|---|---|
| Recall@6 | Share of the question's companies with ≥1 chunk from an accepted Item in the top 6 | no |
| Hit@1, MRR | Rank of the first relevant chunk | no |
| Faithfulness | Judge lists the answer's claims; each is checked against the excerpts only | yes |
| Answer relevance | Judge's 1–5 rating | yes |
| Correct refusal | Unanswerable questions (unknown company, future, live price) must be refused | yes |
| Citation validity | Deterministic: every `[n]` refers to an excerpt | yes |

Known limitations of the evaluation, stated up front:
- Relevance labels are at the (company, Item) level, not the passage level. A chunk from the right Item
  can still miss the specific fact. Faithfulness and relevance judging cover that gap.
- The judge is the same model family as the generator, which can be lenient toward its own outputs.
  The judge prompt restricts it to the excerpts. A stronger fix is a different judge model or a small
  human-labelled sample.
- 47 questions give coarse estimates; a ±5-point difference between modes is within noise.

## What I would add next

- Table-aware retrieval for numeric questions (index XBRL facts as structured data and answer "what was
  revenue" from the facts, not from text).
- Query decomposition for comparison questions (one sub-query per company, then merge).
- A cache keyed by (question, index version) and streaming responses in the API.
