# rag-v2

Multi-user RAG service: FastAPI + Postgres/pgvector, hybrid retrieval (BM25 + vector, RRF, cross-encoder rerank),
JWT auth, MLflow-tracked retrieval eval (recall@5 on `src/test_sets/`), Docker Compose.

Goal until 20 Nov 2026: tests + a CI eval gate, deploy to Azure with Terraform (Key Vault, managed identity),
OpenTelemetry tracing and a dashboard, a nightly eval against production.

## How we work: three levels

I'm on a one-month plan. I don't hand-write everything; I own the decisions and can explain every line.

**1. I write it myself (use the learning loop below).** Small, but it is the judgment an interviewer will probe:
- the recall@5 threshold and the bootstrap confidence-interval check in the CI eval job
- what the alerts fire on and at what level
- anything that decides whether retrieval quality is "good enough"

**2. You draft, I own it.** GitHub Actions YAML, Terraform, Dockerfiles, OpenTelemetry spans, pytest tests.
- Before writing, say in 3-5 lines what you'll build and why.
- After writing, explain each block: what it does and what breaks without it.
- Leave one small change for me to make by hand, and tell me which.

**3. Delegate.** UI tweaks in `src/static/`, README formatting, boilerplate config. Just do it and tell me what changed.

If you're unsure which level something is, ask.

## Learning loop (level 1 only)

1. I explain the task in my own words; you push back on gaps before any code.
2. I write a draft; you don't offer your version first.
3. You review it against my intent: logic errors vs syntax I don't know yet. Explain the fix, scoped to the broken piece.
4. I rewrite from the explanation, not by copy-paste.
5. Add what I missed to `JOURNAL.md` under "syntax I don't own yet"; point out repeats.

## Rules

- No rabbit holes: solve today's task only. If I've been stuck 45 minutes, suggest the simplest workaround.
  Interesting-but-not-needed ideas go to `LATER.md`, not into the code.
- Every number in the README must come from a run I can reproduce (MLflow run or CI log). Don't write numbers you didn't measure.
- Don't change retrieval behaviour (chunking, top_k, rerank, BM25) without rerunning `python src/track_eval.py`
  from the repo root and reporting recall@5 before and after.
- Never put secrets in code or committed files. Local: `.env` (gitignored). Azure: Key Vault.
- Interview prep mode: if I say "interview prep", switch off all of this and don't help.