# RAG v3 — Measured, Eval-Driven Document Q&A

A multi-user Retrieval-Augmented Generation (RAG) service built with an eval-first discipline:
every retrieval change is judged by a number on a hand-written eval set, never on vibes. Users
register, upload documents, and chat with them in a browser UI; conversations are stored in
Postgres and follow-up questions work. It runs as a FastAPI service in Docker, stores documents
in Postgres + pgvector, and tracks every retrieval experiment in MLflow. Every push and pull
request runs the unit tests and a retrieval eval in GitHub Actions, and the eval fails the build
if recall@5 drops below a threshold. This README also records what didn't work and what is still
open.

## Architecture

```
upload:  file -> load_document -> chunk (200 tokens, 50 overlap) -> embed (bge-large, 1024-d)
              -> Postgres (chunks + pgvector), per user_id, duplicate uploads skipped (SHA-256)
ask:     question (+ last 3 turns of the chat, loaded by the server)
              -> rewrite follow-ups into a standalone question (LLM call, skipped for a new chat)
              -> embed -> BM25 over the user's chunks + pgvector search
              -> RRF fusion -> cross-encoder rerank -> top 5 chunks -> LLM answer
              -> user + assistant messages saved to the conversation
eval:    retrieval recall@5 through the same Postgres code path; every config logged to MLflow
CI:      on every push and PR: unit tests, then a fresh Postgres + pgvector service container,
              seed the eval document, run recall@5 on the production config, report it with a
              95% bootstrap interval, fail below the threshold
```

### Code map

- `src/api.py` — FastAPI app and routes (below).
- `src/auth.py` — bcrypt password hashing, JWT creation and the `get_current_user` dependency.
- `src/db.py` — Postgres access: schema creation at startup, connection pool, per-user pgvector
  search, documents, and the conversation/message helpers.
- `src/pipelines.py` — orchestration: upload processing, `answer_question` (rewrite, then
  `ask_question_phase1` retrieval with `use_bm25` / `use_rerank` / `top_k` switches, then
  `ask_question_phase2` LLM), and the recall@5 eval loop.
- `src/embedder.py` — dense embeddings (`BAAI/bge-large-en-v1.5`), BM25, Reciprocal Rank Fusion,
  cross-encoder reranker (`BAAI/bge-reranker-base`).
- `src/model.py` — LLM layer: `LLM_BACKEND` = `ollama` | `claude` | `gemini`. Gemini calls (answer
  and question rewrite) retry each model with a growing pause, then fall back to the next model.
- `src/static/index.html` — the single-page UI (no framework): sign in / create account, upload,
  document picker, chat, conversation sidebar. Served by FastAPI at `/`.
- `src/track_eval.py` — runs the retrieval eval. Without flags: the full a–e grid, each config
  logged as an MLflow run. With `--gate` (used in CI): only the production config, a 95% bootstrap
  interval written to the job summary, and exit code 1 if recall@5 is below the threshold.
- `src/seed_eval.py` — creates the schema and an eval user in an empty database, uploads the eval
  document through the same `processing_file_uploads` path as real uploads, and prints the user id.
- `src/check_overlap.py` — checks a draft of new eval questions: flags any question that shares
  3+ consecutive words with the paper, and any `source_snippet` not found in the PDF text.
- `src/test_sets/` — the eval document (`wf.pdf`) and its questions (`eval_set_one.json`, one
  JSON object per line), kept side by side so anyone can reproduce the eval.
- `src/timing.py` — per-stage timings written to the log (logger `rag.timing`).
- `test_testing/` — pytest unit tests (chunker, `snippets_hit`, RRF, auth); `conftest.py` sets a
  throwaway `JWT_SECRET` and fakes `sentence_transformers` so the tests run fast and offline.
- `.github/workflows/ci.yml` — the CI workflow: jobs `runn_test` and `runn_eval`.
- `Dockerfile` + `docker-compose.yml` — `api` (port 8003) and `db` (`pgvector/pgvector:pg16`)
  with persistent volumes for Postgres data and the HuggingFace model cache.

### Routes

| Route | What it does |
|---|---|
| `POST /register`, `POST /login` | username + password (8+ chars, bcrypt-hashed); login returns a JWT |
| `POST /upload` | add a document for the signed-in user |
| `GET /documents` | the user's own document list |
| `POST /ask` | `question`, optional `documents_id` (repeat per ticked document), optional `conversation_id` |
| `GET /conversations` | the user's chats, most recently active first |
| `GET /conversations/{id}` | one chat's messages (with sources) |
| `DELETE /conversations/{id}` | delete a chat and its messages |
| `GET /` | the web UI |

Every route except `/`, `/register` and `/login` needs `Authorization: Bearer <token>`.
The `user_id` comes only from the signed token, never from the request body.

### Data model

`users`, `documents`, `chunks` (with `vector(1024)`), `conversations`, `messages`. Messages
reference their conversation with `ON DELETE CASCADE` and are indexed on
`(conversation_id, id)`. Assistant rows store the source line and the retrieved passages as
`JSONB`; user rows store which documents were ticked (`INTEGER[]`).

## Conversations and memory

LLMs are stateless, so "memory" means putting the right text in front of the model.

1. **The server owns the history.** The browser sends only a `conversation_id`. The server
   loads the last 3 turns (6 messages, role + content) from Postgres, so a client can't forge
   history.
2. **Query rewriting.** Retrieval only sees the question text, so "and what is its number?"
   would search for nothing useful. For a follow-up, one small structured LLM call rewrites it
   into a standalone question ("What is the invoice number?"). A new chat skips this call.
3. **The rewritten question is used for everything downstream**: BM25, vector search, rerank
   and the answer prompt. The answer prompt does not get the history. The UI still shows the
   user's original wording.
4. **Ownership.** Another user's `conversation_id` behaves exactly like one that doesn't exist
   (404), so ids can't be probed.
5. **Failure behavior is a decision, not an accident.** If the rewrite (or answer) LLM fails
   after its retries, `/ask` returns 503 with a clean message and saves nothing. I chose this
   over silently searching with the unresolved follow-up, which would give a poor answer.
6. **What gets saved:** the user message and the assistant message, after the answer is ready.
   The "no files uploaded" reply and failed requests are not saved.

Alternatives I compared and didn't build: multi-query expansion with RRF, and agentic
retrieval (the LLM calls a search tool). Rewriting was the cheapest fit for this pipeline.

## Status

| Step | Status |
|---|---|
| FastAPI `/upload` + `/ask` | Done |
| Docker + compose, volumes, persistence verified | Done |
| Postgres + pgvector replaces FAISS + JSON files (pool, duplicate skip, per-user isolation) | Done |
| Provider-agnostic LLM switch with retry/fallback | Done |
| Stage timings in the container log | Done |
| Retrieval eval through Postgres + MLflow experiment tracking | Done |
| Auth (register, login, JWT) and per-user isolation | Done |
| Web UI: login, upload, document picker, chat | Done |
| Conversations + chat memory (sidebar, follow-up questions) | Done |
| Unit tests (pytest) run in CI on every push and PR | Done |
| CI-gated retrieval eval (Postgres service container, threshold, bootstrap interval) | Done |
| Eval set grown from 19 to 44 answerable questions | Done |
| Branch protection (eval required before merging) | Next |
| Cloud deployment (Azure) | Not started (prep next) |
| GPU passthrough for the embedder/reranker | Deferred (not useful for a CPU deployment) |

## Results

### Retrieval quality (MLflow, Postgres + pgvector, 44 answerable questions)

Eval set: `src/test_sets/eval_set_one.json`, written from a nursing-home fall-risk research paper
(`src/test_sets/wf.pdf`). 52 questions; 8 are unanswerable and skipped, leaving 44. recall@5
counts a question as a hit when every required gold snippet appears in the top-5 retrieved
chunks. Each config is one run (`python src/track_eval.py --user-id <id>`), measured 2026-11-03.

| Config | recall@5 | Hits |
|---|---|---|
| a) vector only, top_k 20 | 59.1% | 26/44 |
| b) vector + BM25 (RRF), no rerank, top_k 20 | 65.9% | 29/44 |
| c) vector + BM25 + rerank, top_k 20 (**current**) | **72.7%** | 32/44 |
| d) vector + BM25 + rerank, top_k 10 | 70.5% | 31/44 |
| e) vector + BM25 + rerank, top_k 40 | 68.2% | 30/44 |

**95% bootstrap interval for config c: [59.1%, 86.4%]** (10,000 resamples of the 44 per-question
hits and misses, seeded). CI reports the same interval on every run's Summary page.

- **The original 19 questions still score 18/19; the 25 new ones score 14/25.** Nothing in the
  pipeline changed. The new questions are paraphrased the way a non-expert would ask, and that
  exposed a weakness the original set hid.
- **Most misses are vocabulary mismatch.** The question and the paper use different words for the
  same thing: "gadgets and apps" vs "digital technologies", "managers' backing" vs "leadership
  commitment", "senior nurses with extra qualifications" vs "nurse practitioners". BM25 has no
  shared words to match, and the dense embedder has to bridge every synonym on its own.
- **The ordering a < b < c still holds:** BM25 helps over vector-only, and reranking helps most.
- **top_k 10, 20 and 40 are tied.** They differ by one or two questions, well inside the
  ±13-point interval, so this set can't separate them.
- Caveats: one document, one run per config. Some new questions are paraphrased harder than a
  real user would ask (the generation prompt forbade the paper's own terms), so real-user recall
  is likely somewhere between this result and the original set's.

*Earlier result (2026-10-23, the original 19 questions): config c 94.7% (18/19), vector-only
63.2%, BM25 hybrid 68.4%, top_k 10 78.9%, top_k 40 94.7%. The single miss was the multi-chunk
question "Why does the review argue that current fall risk assessment tools are insufficient,
and what direction does it recommend for future development?", and it still misses.
`docs/mlflow_compare.png` shows that earlier grid. An even earlier in-memory FAISS baseline
scored hit@5 = 90.9% (20/22) with a different denominator, so it is not comparable.*

### Latency per `/ask` query

Measured 2026-10-06, Docker Compose stack on a Windows 11 host, CPU-only embedder and
reranker, one run each (not averaged).

| Stage | Local qwen3.5:9b (Ollama) | Hosted Gemini flash |
|---|---|---|
| rerank (CPU cross-encoder) | 1.2 s | 1.7 s |
| vector search (pgvector) | ~0 s | 0.007 s |
| LLM call | 57.7 s | 5.5 s |
| total | ~59 s | 7.8 s |

The LLM call dominates end-to-end latency; retrieval is cheap. Uploading an 87-chunk document
took ~63 s on CPU (embedding) at the time, a one-time cost per document that I accepted for v1; a
smaller embedding model or background embedding would reduce it. On a GitHub Actions runner the
eval document (96 chunks with the current loader and chunker) takes ~82 s to embed. Hosted
Gemini sends retrieved chunks to Google; Ollama remains the private/local option.

A single later run on a hosted Gemini model (2026-10-20): embed 0.5 s, rerank 2.4 s, LLM 12.4 s,
total 15.4 s. LLM latency varies a lot run to run. The rewrite step adds one more LLM call to
follow-up questions only; its added latency has not been measured yet.

### Generation quality (SQuAD v2, 100 Q, earlier pipeline — historical)

| Backend | Faithfulness | Accuracy | Hallucinated-on-impossible | False-refusal-on-answerable |
|---|---|---|---|---|
| qwen3-coder-30b | 88% | ~67% | 23/55 (42%, ~7% once judge/label noise is separated out) | 6/45 (13%) |
| Claude Sonnet 5 | 96% | 76% | 11/55 (20%, ~7% real fabrication after the same correction) | 9/45 (20%) |

Measured before the move to Postgres. The harness that produced these numbers ran on an
in-memory FAISS index, not the real pipeline, and was removed in the 2026-10-24 cleanup, so
these numbers can't be reproduced from the current code. SQuAD was used to debug the harness,
not to tune retrieval.

## Testing

**Automated (GitHub Actions, `.github/workflows/ci.yml`), on every push and pull request:**

- `runn_test` — pytest unit tests in `test_testing/`: chunk sizes, starts and overlap (including
  the edge case that found the duplicate-tail bug below), `snippets_hit` normalization
  (case, punctuation and PDF hyphenation), RRF ordering and limits, password hashing, and JWT
  expiry and forged-signature rejection.
- `runn_eval` — starts a `pgvector/pgvector:pg16` service container with a health check, seeds
  it with `seed_eval.py`, runs `track_eval.py --gate`, writes recall@5 and its bootstrap interval
  to the run's Summary page, and fails if recall@5 is below 0.65 (the 44-question baseline,
  0.727, minus about three questions).

**By hand**, the conversations feature was tested against a public benchmark document (llmware
`rag_instruct_benchmark_tester`, 200 passages loaded as one file):

- **Answers and refusals:** an answerable question returned the right value with its source;
  a question whose answer isn't in the document was refused.
- **Memory:** a follow-up with a pronoun ("and what is its number?") resolved correctly, both in
  the UI and through the API.
- **Persistence:** refresh and hard-refresh keep the user signed in and reopen the chat;
  sign-out then sign-in lists the chats.
- **Isolation:** a second account sees no chats or documents of the first; its `conversation_id`
  requests return 404; no token returns 401; a made-up id returns 404.
- **Document selection:** with only a second document ticked, a question about the first was
  refused; a question about the second was answered.
- **Edge cases:** empty and whitespace-only input, deleting the open chat, two tabs with a
  chat deleted in one, long multi-paragraph answers, and a real LLM rate-limit (Gemini free-tier
  429) which produced the clean 503 and saved nothing.

## Key findings

- **A small, self-written eval set can flatter the system.** The original 19 questions largely
  reused the paper's wording and scored 94.7%. Adding 25 questions phrased in everyday words
  brought config c to 72.7%, with no code change. The new questions were checked by a script
  (`check_overlap.py`) for copied phrases and for snippets that exist in the PDF, then reviewed
  by hand.
- **Report the interval, not just the point.** With 44 questions, each one is worth about
  2.3 points, and the 95% bootstrap interval for config c is [59.1%, 86.4%]. Without it, the
  top_k grid looks like it picks a winner; with it, top_k 10, 20 and 40 are visibly tied.
- **Tests find bugs that metrics hide.** An edge-case unit test (a document of exactly 200
  tokens) showed the chunker emitted a second chunk made entirely of tokens already in the
  first. The fix stops once a chunk reaches the end of the document; recall@5 on the original
  set was unchanged (18/19) after re-chunking.
- **A metric bug can look exactly like a real result.** A whitespace mismatch between
  PDF-extracted text and hand-typed snippets gave a 0% hit rate, then a strict substring
  match still under-counted (hyphenation, curly quotes). The fix was a normalized comparison
  (`snippets_hit`: lowercase letters and digits only, both sides). Same retrieval, hit@5 rose
  from ~27-36% to 81.8% at the time.
- **An unbounded candidate list is not "hit@5".** With the reranker off, an early eval passed
  the whole fused list to the hit check and reported 100%. Truncated to 5, it was 72.7%.
- **"Hallucination on impossible questions" needed splitting by faithfulness.** Most such
  answers were grounded in the context and only disagreed with an overly strict label; the
  naive metric overstated hallucination ~3x.
- **CI catches "works on my machine".** The first CI eval run failed because `mlflow` was
  installed in the local venv but missing from `requirements.txt`. A later run failed because a
  direct push to `master` silently put the threshold back from 0.65 to 0.85 — a change a pull
  request diff would have shown.
- **Moving state into Postgres broke the eval harness** (it read JSON files from disk). It now
  reads through the psycopg pool and uses the same retrieval code as the API.
- **Relative paths depend on where you launch from.** Running the eval from `src/` created a
  second `mlflow.db` there, and running it from the root couldn't find the eval set. The
  eval-set path is now anchored on the file; the MLflow store still follows the working
  directory, so run from the repo root.
- **UI state vs server state.** A new chat created while the user clicked another chat didn't
  appear in the sidebar until a reload: the refresh was guarded by "am I still on the chat I
  asked in?". Fix: always refresh the list when the server created a chat, and guard only the
  decision to switch to it.
- **Free-tier LLM quotas are a real failure mode.** The Gemini free tier (20 requests per day per
  model) ran out during testing. The answer step already fell back across models; the rewrite
  step didn't, so every follow-up failed. It now uses the same retry and fallback.
- Two earlier conclusions (a "31.6% reranker-on-everything" result and "chunk boundaries
  destroy ~25% of misses") were drawn with the under-counting matcher and are withdrawn.
- Contextual Retrieval (LLM-written per-chunk blurbs) was implemented earlier, but its effect
  on recall was never cleanly measured.

## Limitations

- **Small eval set, single document.** 44 answerable questions on one academic paper; results
  may not generalize to other document types. Some new questions are paraphrased harder than
  real users would ask. Follow-up retrieval with and without the rewrite step has not been
  evaluated.
- **No held-out test set.** All 44 questions are used for reporting; tuning retrieval against
  them would overfit. A separate, untouched set is needed before any tuning.
- **Misses are not yet diagnosed per question.** It isn't known yet whether the gold chunk for
  each miss falls outside the top 20 candidates (a first-stage problem) or is dropped by the
  reranker.
- **Chunking is naive.** Fixed 200-token windows with 50 overlap, no sentence or paragraph
  awareness. Its measured impact is unknown.
- **PDF and DOCX extraction artifacts.** Line-break hyphenation hurts BM25 tokenization. DOCX
  table text is appended after the paragraphs, flattened, and merged cells can repeat, so a
  label in a table (e.g. "Week 9") can end up far from the entries it describes.
- **CPU-only embedder and reranker** in the container; uploads are slow (~63 s for 87 chunks
  locally, ~82 s on a CI runner).
- **Heavy image and slow CI.** `torch` from PyPI on Linux is the CUDA build (several GB), which
  makes the Docker image large and the CI eval job take 6–10 minutes, mostly downloads. The
  Hugging Face models are not cached between CI runs.
- **`timing.py` logs to stdout**, so CI captures the seed script's user id with `tail -n 1`.
- **Memory is query rewriting only.** The answer prompt doesn't see the chat, history is capped
  by turn count (3) but not by length, and a rewrite that is wrong sends retrieval the wrong way.
- **One request at a time in the UI.** Send is disabled while any answer is loading, and tabs
  don't sync live (a chat created in one tab shows in another after a refresh).
- **A failed LLM call fails the request** (503) rather than degrading; chosen on purpose.
- **Latency numbers are single runs** and the exact Gemini model id was not recorded.
- **BM25 candidate count is fixed** by `hybdrid_embedding_top_k` in `config.py`; the `top_k`
  switch only changes the vector search and the fused list.
- **No current generation eval.** The SQuAD numbers above are historical; answer quality on the
  real document through the Postgres pipeline has not been measured.
- **Security hardening before a public deploy:** the compose file publishes Postgres on port
  5432, the app connects as the `postgres` superuser, and secrets live in a local `.env`.
  There is no rate limiting, and the delete confirmation uses the browser's built-in dialog.
- **Not yet deployed.** `master` isn't protected yet, so a red check doesn't block a merge.

## Setup and running

1. Create your config file from the template, then fill it in:

   ```bash
   cp .env.example .env             # Windows PowerShell: Copy-Item .env.example .env
   ```

   | Variable | Needed | Meaning |
   |---|---|---|
   | `POSTGRES_PASSWORD` | always | password for the Postgres container and the app's connection |
   | `JWT_SECRET` | always | long random string that signs login tokens (the app won't start without it) |
   | `LLM_BACKEND` | always | `gemini`, `claude` or `ollama` (defaults to `ollama` if unset) |
   | `GEMINI_API_KEY`, `GEMINI_MODELS` | if `gemini` | key and a comma-separated list of models, tried in order |
   | `ANTHROPIC_API_KEY` (`CLAUDE_MODEL` optional) | if `claude` | key for the Claude backend |
   | `OLLAMA_BASE_URL` | if `ollama` | where Ollama listens; inside Docker use `http://host.docker.internal:11434/v1` |

   `POSTGRES_HOST` is set to `db` by `docker-compose.yml`; you don't put it in `.env`.

2. Start everything (the first build downloads the embedding and reranker models, so it takes a while):

   ```bash
   docker compose up --build        # UI + API on http://127.0.0.1:8003, Postgres on 5432
   ```

3. Open http://127.0.0.1:8003, create an account, upload a document, and ask a question.

Use `127.0.0.1`, not `localhost`, on Windows (IPv6 resolution hangs). See `src/config.py` for the
chunking and retrieval parameters. The local Ollama option needs Ollama running on your machine
with the `qwen3.5:9b` model pulled.

### Running the retrieval eval locally

Only the database is needed, not the API. From the repo root:

```bash
docker compose up -d db                              # Postgres only
python src/seed_eval.py                              # once: creates the eval user, uploads wf.pdf, prints the user id
python src/track_eval.py --user-id <id>              # the full a-e grid, logged to MLflow
python src/track_eval.py --user-id <id> --gate       # what CI runs: config c only, fails below the threshold
mlflow ui --port 5000                                # open http://127.0.0.1:5000
```

Run `seed_eval.py` once per database: a second run fails because the eval user already exists, so
reuse the id it printed. Each config's missed questions are logged as the MLflow artifact
`missed_questions.json`.

### Running the unit tests

```bash
pytest                                               # pytest.ini points at test_testing/ and src/
```

## Roadmap

1. Branch protection on `master`: require pull requests and the `runn_test` and `runn_eval`
   checks before merging.
2. Deployment prep: stop publishing the Postgres port, a least-privilege database role,
   secrets from the cloud's secret store, a hosted LLM backend (the local Ollama option can't
   be reached from the cloud).
3. Cloud deployment (Azure): container registry, container app, managed Postgres.
4. Diagnose the 12 misses: the rank of each gold chunk in the vector top 20, the BM25 top 20,
   and after reranking. Then try query expansion (e.g. HyDE) against the vocabulary gap, on a
   held-out set to avoid overfitting.
5. CI speed and size: CPU-only torch in the Dockerfile and CI, cache the Hugging Face models
   between runs, split `requirements.txt` into app and dev (`pytest`, `mlflow`), and move
   `timing.py` logs to stderr.
6. Follow-up retrieval eval (rewrite on vs off) logged to MLflow.
7. Polish: in-app delete dialog, per-chat pending state, DOCX table handling.
8. Optional: GPU passthrough for the embedder and reranker as a Docker-for-ML exercise.