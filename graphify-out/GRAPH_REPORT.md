# Graph Report - rag-v2  (2026-10-01)

## Corpus Check
- 17 files · ~47,418 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 95 nodes · 228 edges · 10 communities
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0d9d6bee`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pipelines.py
- api.py
- RAG v3 — Measured, Eval-Driven Document Q&A
- PDF/DOCX Document Loaders
- model.py
- Graphify Knowledge Graph
- embedder.py
- RAG Retrieval Pipeline
- manual_initialization_pipeline

## God Nodes (most connected - your core abstractions)
1. `load_document()` - 12 edges
2. `RAG v3 — Measured, Eval-Driven Document Q&A` - 10 edges
3. `manual_initialization_pipeline()` - 9 edges
4. `answer_question()` - 9 edges
5. `evaluate_recall_at_5()` - 8 edges
6. `chunk()` - 7 edges
7. `define_LLM_EVALUATION_SCHEMA()` - 7 edges
8. `enhancing_chunks()` - 7 edges
9. `build_index()` - 7 edges
10. `processing_file_uploads()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Checking_claude()` --indirect_call--> `askQuestionToClaude()`  [INFERRED]
  src/pipelines.py → src/model.py
- `Checking_claude()` --indirect_call--> `ask_CLAUDE_TO_EVALUATE_RESPONSE()`  [INFERRED]
  src/pipelines.py → src/model.py
- `manual_initialization_pipeline()` --calls--> `enhancing_chunks()`  [INFERRED]
  src/pipelines.py → src/model.py
- `ask_question()` --calls--> `answer_question()`  [EXTRACTED]
  src/api.py → src/pipelines.py
- `build_index()` --calls--> `chunk()`  [EXTRACTED]
  src/pipelines.py → src/chunker.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Navigation Tooling** — claude_graphify_query_commands, claude_graphify_wiki_index, claude_graph_report, claude_graphify_knowledge_graph [EXTRACTED 0.75]
- **RAG Ingest-to-Answer Flow** — requirements_document_loaders, requirements_sentence_transformers_embeddings, requirements_faiss_vector_index, requirements_openai_llm_backend [INFERRED 0.75]

## Communities (10 total, 0 thin omitted)

### Community 0 - "pipelines.py"
Cohesion: 0.24
Nodes (18): bm25_search(), embed_question(), reranker(), RFF_TOP_PICKS(), evaluating_embedding_model(), reading_the_golden_set(), answer_question(), askQuestionToIndex() (+10 more)

### Community 1 - "api.py"
Cohesion: 0.24
Nodes (13): FastAPI, lifespan(), upload_file(), load_document(), pick_files(), read_chunk_history(), read_docx(), read_pdf() (+5 more)

### Community 2 - "RAG v3 — Measured, Eval-Driven Document Q&A"
Cohesion: 0.15
Nodes (12): Architecture, Evaluation methodology, Generation quality (SQuAD, 100 Q, qwen3-coder-30b vs. Claude Sonnet 5), Key findings, Limitations, RAG v3 — Measured, Eval-Driven Document Q&A, Results, Retrieval quality (real eval set, 22 Q, hit@5 — strict: every required snippet must be (+4 more)

### Community 3 - "PDF/DOCX Document Loaders"
Cohesion: 0.33
Nodes (6): PDF/DOCX Document Loaders, FAISS Vector Index, Ollama Local LLM Backend, OpenAI LLM Backend, Sentence-Transformers Embedding Model, Tiktoken Token Counting

### Community 4 - "model.py"
Cohesion: 0.37
Nodes (12): ask_AI_TO_EVALUTE_RESPONSE(), ask_CLAUDE_TO_EVALUATE_RESPONSE(), askQuestionToAI(), askQuestionToClaude(), build_judge_user_prompt(), _claude_structured_call(), define_LLM_EVALUATION_SCHEMA(), enhancing_chunks() (+4 more)

### Community 5 - "Graphify Knowledge Graph"
Cohesion: 0.50
Nodes (5): GRAPH_REPORT.md, Graphify Knowledge Graph, Graphify Query/Path/Explain Commands, Graphify AST-only Update, Graphify Wiki Index

### Community 6 - "embedder.py"
Cohesion: 0.31
Nodes (8): chunk(), read_chunks(), save_chunks(), save_user_chunk_mapping(), add_chunks_to_index(), read_embedding_index(), save_embedding_index(), automatic_initialization_pipeline()

### Community 7 - "RAG Retrieval Pipeline"
Cohesion: 0.67
Nodes (3): HuggingFace Datasets / scikit-learn Evaluation, RAG Retrieval Pipeline, Typer/Rich CLI Framework

### Community 8 - "manual_initialization_pipeline"
Cohesion: 0.38
Nodes (7): post, ask_question(), build_bm25_index(), fais_chunks_embedder(), populate_index(), build_index(), manual_initialization_pipeline()

## Knowledge Gaps
- **15 isolated node(s):** `Architecture`, `What's implemented`, `Evaluation methodology`, `Generation quality (SQuAD, 100 Q, qwen3-coder-30b vs. Claude Sonnet 5)`, `Retrieval quality (real eval set, 22 Q, hit@5 — strict: every required snippet must be` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_document()` connect `api.py` to `pipelines.py`, `manual_initialization_pipeline`, `model.py`, `embedder.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `upload_file()` connect `api.py` to `manual_initialization_pipeline`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `processing_file_uploads()` connect `api.py` to `pipelines.py`, `manual_initialization_pipeline`, `embedder.py`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **What connects `Architecture`, `What's implemented`, `Evaluation methodology` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._