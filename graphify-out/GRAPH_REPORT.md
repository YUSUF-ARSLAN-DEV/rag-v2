# Graph Report - rag-v2  (2026-08-30)

## Corpus Check
- 12 files · ~167,312 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 43 nodes · 72 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.77)
- Token cost: 35,000 input · 3,844 output

## Community Hubs (Navigation)
- LLM Answering & Evaluation
- RAG Dependency Stack
- Chunking & Config
- Graphify Tooling Docs
- Document Loading
- Embedding & Index Building
- Persistence Load Path
- Embedding Model Evaluation
- CLI & Evaluation Deps
- Query Embedding Path

## God Nodes (most connected - your core abstractions)
1. `load_document()` - 6 edges
2. `manual_initialization_pipeline()` - 5 edges
3. `askQuestionToAI()` - 5 edges
4. `reading_the_golden_set()` - 4 edges
5. `main()` - 4 edges
6. `ask_AI_TO_EVALUTE_RESPONSE()` - 4 edges
7. `Graphify Knowledge Graph` - 4 edges
8. `chunk()` - 3 edges
9. `read_chunks()` - 3 edges
10. `embed_chunks()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `manual_initialization_pipeline()` --calls--> `chunk()`  [EXTRACTED]
  main.py → chunker.py
- `manual_initialization_pipeline()` --calls--> `load_document()`  [EXTRACTED]
  main.py → document_loader.py
- `main()` --calls--> `reading_the_golden_set()`  [EXTRACTED]
  main.py → evaluate.py
- `automatic_initialization_pipeline()` --calls--> `read_chunks()`  [EXTRACTED]
  main.py → chunker.py
- `manual_initialization_pipeline()` --calls--> `embed_chunks()`  [EXTRACTED]
  main.py → embedder.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **RAG Ingest-to-Answer Flow** — requirements_document_loaders, requirements_sentence_transformers_embeddings, requirements_faiss_vector_index, requirements_openai_llm_backend [INFERRED 0.75]
- **Graphify Navigation Tooling** — claude_graphify_query_commands, claude_graphify_wiki_index, claude_graph_report, claude_graphify_knowledge_graph [EXTRACTED 0.75]

## Communities (10 total, 3 thin omitted)

### Community 0 - "LLM Answering & Evaluation"
Cohesion: 0.60
Nodes (5): main(), ask_AI_TO_EVALUTE_RESPONSE(), askQuestionToAI(), define_LLM_EVALUATION_SCHEMA(), get_client()

### Community 1 - "RAG Dependency Stack"
Cohesion: 0.33
Nodes (6): PDF/DOCX Document Loaders, FAISS Vector Index, Ollama Local LLM Backend, OpenAI LLM Backend, Sentence-Transformers Embedding Model, Tiktoken Token Counting

### Community 3 - "Graphify Tooling Docs"
Cohesion: 0.50
Nodes (5): GRAPH_REPORT.md, Graphify Knowledge Graph, Graphify Query/Path/Explain Commands, Graphify AST-only Update, Graphify Wiki Index

### Community 4 - "Document Loading"
Cohesion: 0.70
Nodes (4): load_document(), read_docx(), read_pdf(), read_txt()

### Community 5 - "Embedding & Index Building"
Cohesion: 0.50
Nodes (4): embed_chunks(), populate_index(), save_embedding_index(), manual_initialization_pipeline()

### Community 6 - "Persistence Load Path"
Cohesion: 0.67
Nodes (3): read_chunks(), read_embedding_index(), automatic_initialization_pipeline()

### Community 8 - "CLI & Evaluation Deps"
Cohesion: 0.67
Nodes (3): HuggingFace Datasets / scikit-learn Evaluation, RAG Retrieval Pipeline, Typer/Rich CLI Framework

## Knowledge Gaps
- **5 isolated node(s):** `Graphify Wiki Index`, `Ollama Local LLM Backend`, `Tiktoken Token Counting`, `Typer/Rich CLI Framework`, `HuggingFace Datasets / scikit-learn Evaluation`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_document()` connect `Document Loading` to `Chunking & Config`, `Embedding & Index Building`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `askQuestionToAI()` connect `LLM Answering & Evaluation` to `Chunking & Config`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Why does `ask_AI_TO_EVALUTE_RESPONSE()` connect `LLM Answering & Evaluation` to `Chunking & Config`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `Graphify Wiki Index`, `Ollama Local LLM Backend`, `Tiktoken Token Counting` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._