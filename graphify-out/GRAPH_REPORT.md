# Graph Report - rag-v2  (2026-09-12)

## Corpus Check
- 11 files · ~168,667 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 50 nodes · 92 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cf0a4b15`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- model.py
- reading_the_golden_set
- embedder.py
- PDF/DOCX Document Loaders
- main.py
- Graphify Knowledge Graph
- load_document
- RAG Retrieval Pipeline
- read_chunks
- embed_question

## God Nodes (most connected - your core abstractions)
1. `load_document()` - 6 edges
2. `askQuestionToClaude()` - 6 edges
3. `ask_CLAUDE_TO_EVALUATE_RESPONSE()` - 6 edges
4. `reading_the_golden_set()` - 5 edges
5. `manual_initialization_pipeline()` - 5 edges
6. `Checking_claude()` - 5 edges
7. `askQuestionToAI()` - 5 edges
8. `ask_AI_TO_EVALUTE_RESPONSE()` - 5 edges
9. `define_LLM_EVALUATION_SCHEMA()` - 5 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `manual_initialization_pipeline()` --calls--> `chunk()`  [EXTRACTED]
  main.py → chunker.py
- `manual_initialization_pipeline()` --calls--> `load_document()`  [EXTRACTED]
  main.py → document_loader.py
- `main()` --calls--> `reading_the_golden_set()`  [EXTRACTED]
  main.py → evaluate.py
- `Checking_claude()` --calls--> `ask_CLAUDE_TO_EVALUATE_RESPONSE()`  [EXTRACTED]
  main.py → model.py
- `Checking_claude()` --calls--> `askQuestionToClaude()`  [EXTRACTED]
  main.py → model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Navigation Tooling** — claude_graphify_query_commands, claude_graphify_wiki_index, claude_graph_report, claude_graphify_knowledge_graph [EXTRACTED 0.75]
- **RAG Ingest-to-Answer Flow** — requirements_document_loaders, requirements_sentence_transformers_embeddings, requirements_faiss_vector_index, requirements_openai_llm_backend [INFERRED 0.75]

## Communities (10 total, 2 thin omitted)

### Community 0 - "model.py"
Cohesion: 0.38
Nodes (10): main(), ask_AI_TO_EVALUTE_RESPONSE(), ask_CLAUDE_TO_EVALUATE_RESPONSE(), askQuestionToAI(), askQuestionToClaude(), build_judge_user_prompt(), _claude_structured_call(), define_LLM_EVALUATION_SCHEMA() (+2 more)

### Community 1 - "reading_the_golden_set"
Cohesion: 0.50
Nodes (4): evaluating_embedding_model(), reading_the_golden_set(), Checking_claude(), diagnose_hallucinations()

### Community 2 - "embedder.py"
Cohesion: 0.50
Nodes (4): embed_chunks(), populate_index(), save_embedding_index(), manual_initialization_pipeline()

### Community 3 - "PDF/DOCX Document Loaders"
Cohesion: 0.33
Nodes (6): PDF/DOCX Document Loaders, FAISS Vector Index, Ollama Local LLM Backend, OpenAI LLM Backend, Sentence-Transformers Embedding Model, Tiktoken Token Counting

### Community 5 - "Graphify Knowledge Graph"
Cohesion: 0.50
Nodes (5): GRAPH_REPORT.md, Graphify Knowledge Graph, Graphify Query/Path/Explain Commands, Graphify AST-only Update, Graphify Wiki Index

### Community 6 - "load_document"
Cohesion: 0.70
Nodes (4): load_document(), read_docx(), read_pdf(), read_txt()

### Community 7 - "RAG Retrieval Pipeline"
Cohesion: 0.67
Nodes (3): HuggingFace Datasets / scikit-learn Evaluation, RAG Retrieval Pipeline, Typer/Rich CLI Framework

### Community 8 - "read_chunks"
Cohesion: 0.67
Nodes (3): read_chunks(), read_embedding_index(), automatic_initialization_pipeline()

## Knowledge Gaps
- **5 isolated node(s):** `Ollama Local LLM Backend`, `Tiktoken Token Counting`, `Graphify Wiki Index`, `HuggingFace Datasets / scikit-learn Evaluation`, `Typer/Rich CLI Framework`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_document()` connect `load_document` to `embedder.py`, `main.py`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `askQuestionToClaude()` connect `model.py` to `reading_the_golden_set`, `main.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `ask_CLAUDE_TO_EVALUATE_RESPONSE()` connect `model.py` to `reading_the_golden_set`, `main.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **What connects `Ollama Local LLM Backend`, `Tiktoken Token Counting`, `Graphify Wiki Index` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._