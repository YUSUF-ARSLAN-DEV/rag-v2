# RAG v3 — Measured, Eval-Driven Document Q&A

A Retrieval-Augmented Generation (RAG) pipeline built with an eval-first discipline: every
retrieval or prompting change is only kept if it measurably improves a scorecard, never on
vibes. This repo documents both what worked and what didn't — including the parts that are
still broken.

## Architecture

```
ingest:  load_document -> chunk (token-window) -> embed (dense + BM25) -> FAISS + BM25 index
query:   question -> embed -> hybrid retrieve (RRF fusion) -> cross-encoder rerank -> LLM answer
eval:    two harnesses - retrieval metrics (recall@k, MRR) and generation metrics
         (faithfulness, correctness, refusal-on-unanswerable), both LLM-judged where needed
```

- `document_loader.py` — loads `.pdf` / `.docx` / `.txt` into raw text.
- `chunker.py` — splits text into fixed-size token windows (tiktoken `cl100k_base`), with overlap.
- `embedder.py` — dense embeddings (`BAAI/bge-large-en-v1.5`), BM25 keyword index, Reciprocal
  Rank Fusion (`RFF_TOP_PICKS`), and a cross-encoder reranker (`BAAI/bge-reranker-base`).
- `model.py` — the LLM layer: an OpenAI-compatible backend (local Ollama or a hosted
  OpenAI-compatible server) and a Claude (Anthropic API) backend, structurally interchangeable.
- `pipelines.py` — orchestration: document ingestion, the generation eval loop, chunk
  enrichment (Contextual Retrieval).
- `evaluate.py` / `main.py` — retrieval and generation eval entry points.

## What's implemented

| Phase | Status |
|---|---|
| 0 — Foundation (config, provider-agnostic LLM layer, chunk identity) | Done |
| 1 — Persistence (save/load FAISS index + chunks) | Done |
| 2 — Evaluation harness (retrieval: recall@1/@5, MRR; generation: faithfulness, correctness, refusal) | Done |
| 3 — Retrieval upgrades: hybrid search (BM25 + dense, RRF fusion), cross-encoder reranking, Contextual Retrieval | Done, mixed results (see below) |
| 3 — Sentence/paragraph-aware chunking | Not done (see Limitations) |
| 3.5 onward — CI-gated eval, intent routing, API, frontend, deployment | Not started |

## Evaluation methodology

Two separate eval sets, used for different purposes:

1. **SQuAD v2 (`squad_gold.json`, 100 Q)** — scaffolding used early on to debug the harness
   itself (retrieval metrics, LLM-as-judge, refusal tracking). Not used for tuning retrieval —
   SQuAD's "impossible" labels are calibrated to one exact gold paragraph and don't map cleanly
   onto an arbitrary retrieved context, which was confirmed directly during this project (see
   Limitations).
2. **`eval_set_one.json` (22 Q, hand-written)** — a real eval set built from an actual document
   (`sample_sources/healthcare-sample-research-paper.pdf`, a nursing-home fall-risk research
   paper), covering `fact`, `exact_token`, `multi_chunk`, and `impossible` question types. Every
   answerable row's `expected_answer` is traced back to an exact quote (`source_snippet`) in the
   source document. This is the set Phase 3 retrieval changes are measured against.

## Results

### Generation quality (SQuAD, 100 Q, qwen3-coder-30b vs. Claude Sonnet 5)

| Backend | Faithfulness | Accuracy | Hallucinated-on-impossible | False-refusal-on-answerable |
|---|---|---|---|---|
| qwen3-coder-30b | 88% | ~67% | 23/55 (42%, but ~7% once judge/label noise is separated out) | 6/45 (13%) |
| Claude Sonnet 5 | 96% | 76% | 11/55 (20%, ~7% real fabrication after the same correction) | 9/45 (20%) |

Claude showed meaningfully better false-premise detection at the cost of being more
conservative (more false refusals). See "Key findings" below for why the raw hallucination
numbers needed correction before they were trustworthy.

### Retrieval quality (real eval set, 22 Q, hit@5 — strict: every required snippet must be
in the top-5)

Measured with the normalized matcher (`snippets_hit`, see Key findings). Every one of the 22
gold snippets exists in the chunk set, so 100% is reachable.

| Configuration | hit@5 |
|---|---|
| Dense-only, top-5, no rerank | 63.6% (14/22) |
| Dense-only, top-20 → rerank → 5 | 72.7% (16/22) |
| Dense-only, top-50 → rerank → 5 | 77.3% (17/22) |
| Hybrid (BM25 + dense, RRF fusion), top-5 straight from RRF, no rerank | 72.7% (16/22) |
| Hybrid + cross-encoder rerank (RRF top-20 → top-5; reranking the full fused list gives the same 81.8%) | 81.8% (18/22) |
| Hybrid, k=20 per retriever (fused pool up to 40) + rerank → top-5 — **current config** | **90.9%** (20/22) |
| *Reference:* RRF top-20 shortlist, no rerank (ceiling for the reranker) | 86.4% (19/22) |
| + Contextual Retrieval (LLM-generated per-chunk blurb prepended before embedding) | Not re-measured with the corrected matcher |

Each stage helps a little: hybrid over dense (+9 points at top-5, +9 with rerank), rerank over
no rerank (+9 points), and a larger candidate pool helps the reranker (dense: top-5 pool
63.6% → top-20 72.7% → top-50 77.3%). Reranking adds +2 questions over raw RRF order, and the 4 remaining misses all
had the gold chunk inside the top-20 shortlist, so the shortlist is not the bottleneck. With
22 questions (each ~4.5 points) the +2 is suggestive, not conclusive.

*Historical numbers (18.8% dense-only, 36.4% hybrid + rerank) were produced with an
exact-substring matcher that under-counted; they are superseded by the table above.*

## Key findings

- **A metric bug can look exactly like a real result.** Early in generation eval, a
  whitespace mismatch between PDF-extracted text (`\n` at every line-wrap) and plainly-
  transcribed eval snippets (plain spaces) caused a **0% hit rate across every question**,
  which looked like a retrieval failure but was actually a string-comparison bug. Fixed by
  normalizing whitespace at the PDF-extraction source (`document_loader.read_pdf`).
- **The measuring stick was the biggest bug, twice.** After the whitespace fix, hit@5 still
  read 27–36% because `snippet in chunk_text` demanded a byte-exact match, while the
  hand-typed snippets differed from the PDF's raw text in small ways: line-break hyphenation
  (`moni- toring`), curly vs straight quotes, LaTeX markup and dropped hyphens typed into the
  snippet. A diagnostic showed 8/22 snippets exact-matched any chunk; with a normalized
  comparison (`snippets_hit`: lowercase, letters and digits only, applied to both sides,
  stored text untouched) all 22 do. Same retrieval, hit@5 rose from ~27–36% to 81.8%.
- **An unbounded candidate list is not "hit@5".** Turning the reranker off in the eval loop
  passed the *entire* RRF-fused list (65–80 of 96 chunks) to the hit check and reported 100%.
  That was hit@~70, not hit@5. Truncated correctly, the no-rerank number is 72.7%.
- **"Hallucination on impossible questions" needed to be split by faithfulness.** Of the
  cases where a model answered a question the eval set marked impossible, most were
  faithful (context-grounded, just disagreeing with an overly strict SQuAD label);
  only a minority were genuine fabrications. The naive metric overstated hallucination
  by roughly 3x.
- **Two earlier conclusions are withdrawn.** Both were drawn from numbers produced by the
  under-counting matcher: (1) "a reranker given every chunk reached only 31.6%, so the
  bottleneck is chunk text quality", and (2) "chunk boundaries destroy facts in an estimated
  ~25% of remaining misses". The corrected check shows every gold snippet fits inside at
  least one chunk (50-token overlap), so neither claim is supported. They should be
  re-tested with the corrected matcher before being relied on.
- **Contextual Retrieval's true effect on hit@5 was never cleanly measured.** By the time
  it was implemented, the hit-check counting rule had also changed (strict "all snippets
  must match" vs. lenient "any snippet matches"), confounding the before/after comparison.
  The clean, apples-to-apples number was never obtained before the project moved on.

## Limitations

- **Best measured hit@5 is 81.8% (18/22) on a small set.** The 4 misses all had the gold
  chunk in the RRF top-20 but lost it at the rerank stage, so the reranker (and the
  multi-chunk question type, where one required chunk is dropped) is the current weak spot.
  Retrieval hit@5 says the right text was retrieved, not that the LLM answers correctly;
  the generation eval has not yet been run against this real document.
- **Chunking is naive.** `chunker.py` slices the document by raw token count with zero
  sentence/paragraph awareness, and can cut sentences mid-word. Its measured impact is
  currently unknown: the earlier "~25% of misses" estimate is withdrawn (see Key findings).
  This PDF's extracted text also has no reliable paragraph markers (`pypdf` emits a single
  `\n` at both line-wraps and paragraph starts). `pymupdf4llm` is a candidate replacement
  extractor.
- **PDF extraction artifacts.** Line-break hyphenation splits words (`moni- toring`), which
  hurts BM25 tokenization even though an LLM reads it fine. Not yet fixed at the loader.
- **One eval question is questionable.** Row 21 is labeled `impossible` but carries a gold
  snippet, so it should probably be excluded from hit@5.
- **Tiny eval set.** 22 hand-written questions is enough to catch large regressions but too
  small to reliably measure small effects — each question is ~4.5 percentage points.
- **Single test document.** All Phase 3 retrieval tuning was measured against one 14-page
  academic paper, chosen partly because its dense, citation-heavy, jargon-laden prose is a
  genuinely hard, realistic RAG stress test — but results may not generalize to other
  document types (manuals, decks, contracts).
- **LLM-judge noise was never separately quantified.** Faithfulness/correctness scoring
  relies on an LLM judge; the judge's own consistency (would it give the same verdict twice
  on the same input?) was flagged as a needed check early on but never measured.
- **Chunk enrichment (Contextual Retrieval) is not deterministic or persisted.** Blurbs are
  regenerated by an LLM call every time the ingestion pipeline runs, with no fixed seed and
  no caching of the enriched chunks to disk — meaning results vary run-to-run and every
  re-run re-pays the LLM cost.
- **No CI gate, no API, no frontend, no deployment yet** — Phases 3.5 through 7 of the
  project plan (`finalplan.txt`) have not been started.

## Setup

```bash
pip install -r requirements.txt
```

Requires a `.env` with API credentials for whichever backend(s) you use
(`ANTHROPIC_API_KEY` for Claude, `LOCAL_SERVER_*` for a hosted OpenAI-compatible server, or a
local Ollama instance for `ASKLOCAL=True`). See `config.py` for chunking/retrieval parameters.

## Running

- `python main.py` — retrieval eval (hybrid search + rerank) against `eval_set_one.json`.
- `pipelines.main()` / `pipelines.Checking_claude()` — end-to-end generation eval against
  `squad_gold.json`, scoring faithfulness/correctness/refusal.

## Roadmap

See `finalplan.txt` for the full phase-by-phase plan. Immediate next steps: per-query run
logging (Phase 2.5), a clean re-measurement of Contextual Retrieval against the corrected
baseline, evaluating `pymupdf4llm` extraction, then Phase 3.5
(CI-gated eval) and onward to the FastAPI backend and deployment.
