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

| Configuration | hit@5 |
|---|---|
| Dense-only (`bge-large-en-v1.5`) | 18.8% |
| + Hybrid search (BM25 + RRF fusion, top-20) + cross-encoder rerank | 36.4% |
| + Contextual Retrieval (LLM-generated per-chunk blurb prepended before embedding) | Inconclusive (see Limitations) |

Hybrid search **doubled** retrieval recall over dense-only — the clearest, most reliable win
in this project. Reranking and Contextual Retrieval did not produce a further, confirmed
improvement (see below).

## Key findings

- **A metric bug can look exactly like a real result.** Early in generation eval, a
  whitespace mismatch between PDF-extracted text (`\n` at every line-wrap) and plainly-
  transcribed eval snippets (plain spaces) caused a **0% hit rate across every question**,
  which looked like a retrieval failure but was actually a string-comparison bug. Fixed by
  normalizing whitespace at the PDF-extraction source (`document_loader.read_pdf`).
- **"Hallucination on impossible questions" needed to be split by faithfulness.** Of the
  cases where a model answered a question the eval set marked impossible, most were
  faithful (context-grounded, just disagreeing with an overly strict SQuAD label);
  only a minority were genuine fabrications. The naive metric overstated hallucination
  by roughly 3x.
- **A cross-encoder reranker given every chunk, no candidate-selection bottleneck, still
  only reached 31.6% hit@5** — worse than hybrid+rerank's 36.4%. This ruled out "hybrid's
  top-20 shortlist is losing the answer" as the bottleneck and pointed at the chunk text
  itself lacking self-contained signal (generic academic phrasing, citation-list noise).
- **Chunk boundaries directly destroy retrievable facts.** Manual inspection of retrieval
  misses found cases where the correct chunk was retrieved at rank 1, but the chunk was
  truncated mid-sentence or mid-word before the actual answer (e.g., a chunk ending
  `"...The incidence of falls in long-t"` right before the number that answered the
  question). This chunking bug affects an estimated ~25% of remaining misses.
- **Contextual Retrieval's true effect on hit@5 was never cleanly measured.** By the time
  it was implemented, the hit-check counting rule had also changed (strict "all snippets
  must match" vs. lenient "any snippet matches"), confounding the before/after comparison.
  The clean, apples-to-apples number was never obtained before the project moved on.

## Limitations

- **Retrieval accuracy is weak in absolute terms.** Best measured hit@5 on the real eval
  set is 36.4% (strict rule) — meaning most questions do not have all required
  information in the top-5 retrieved chunks. This is the main open problem in the project.
- **Chunking is naive and demonstrably breaks facts.** `chunker.py` slices the document by
  raw token count with zero sentence/paragraph awareness, confirmed to cut sentences and
  even words in half at chunk boundaries. A sentence-aware or paragraph-aware chunker was
  designed but not implemented — deprioritized after estimating it would only recover
  ~25% of remaining misses relative to the engineering effort required, and after
  confirming this PDF's extracted text has no reliable paragraph markers (`pypdf` emits a
  single `\n` at both line-wraps and paragraph starts, indistinguishably).
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

See `finalplan.txt` for the full phase-by-phase plan. Immediate next steps: sentence-aware
chunking, a clean (non-confounded) re-measurement of Contextual Retrieval, then Phase 3.5
(CI-gated eval) and onward to the FastAPI backend and deployment.
