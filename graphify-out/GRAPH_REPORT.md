# Graph Report - rag-v2  (2026-09-12)

## Corpus Check
- 4 files · ~168,667 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 50 nodes · 76 edges · 8 communities
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- LLM Client & Judge Prompts
- Evaluation & Diagnosis Pipeline
- External Libraries & Backends
- Graphify Tooling
- Document Loading
- RAG Eval Dependencies

## God Nodes (most connected - your core abstractions)
1. `askQuestionToClaude()` - 6 edges
2. `ask_CLAUDE_TO_EVALUATE_RESPONSE()` - 6 edges
3. `reading_the_golden_set()` - 5 edges
4. `Checking_claude()` - 5 edges
5. `askQuestionToAI()` - 5 edges
6. `ask_AI_TO_EVALUTE_RESPONSE()` - 5 edges
7. `define_LLM_EVALUATION_SCHEMA()` - 5 edges
8. `load_document()` - 4 edges
9. `Graphify Knowledge Graph` - 4 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `reading_the_golden_set()`  [EXTRACTED]
  main.py → evaluate.py
- `Checking_claude()` --calls--> `ask_CLAUDE_TO_EVALUATE_RESPONSE()`  [EXTRACTED]
  main.py → model.py
- `Checking_claude()` --calls--> `askQuestionToClaude()`  [EXTRACTED]
  main.py → model.py
- `Checking_claude()` --calls--> `reading_the_golden_set()`  [EXTRACTED]
  main.py → evaluate.py
- `main()` --calls--> `ask_AI_TO_EVALUTE_RESPONSE()`  [EXTRACTED]
  main.py → model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Navigation Tooling** — claude_graphify_query_commands, claude_graphify_wiki_index, claude_graph_report, claude_graphify_knowledge_graph [EXTRACTED 0.75]
- **RAG Ingest-to-Answer Flow** — requirements_document_loaders, requirements_sentence_transformers_embeddings, requirements_faiss_vector_index, requirements_openai_llm_backend [INFERRED 0.75]

## Communities (8 total, 0 thin omitted)

### Community 0 - "LLM Client & Judge Prompts"
Cohesion: 0.38
Nodes (10): main(), ask_AI_TO_EVALUTE_RESPONSE(), ask_CLAUDE_TO_EVALUATE_RESPONSE(), askQuestionToAI(), askQuestionToClaude(), build_judge_user_prompt(), _claude_structured_call(), define_LLM_EVALUATION_SCHEMA() (+2 more)

### Community 1 - "Evaluation & Diagnosis Pipeline"
Cohesion: 0.36
Nodes (4): evaluating_embedding_model(), reading_the_golden_set(), Checking_claude(), diagnose_hallucinations()

### Community 3 - "External Libraries & Backends"
Cohesion: 0.33
Nodes (6): PDF/DOCX Document Loaders, FAISS Vector Index, Ollama Local LLM Backend, OpenAI LLM Backend, Sentence-Transformers Embedding Model, Tiktoken Token Counting

### Community 5 - "Graphify Tooling"
Cohesion: 0.50
Nodes (5): GRAPH_REPORT.md, Graphify Knowledge Graph, Graphify Query/Path/Explain Commands, Graphify AST-only Update, Graphify Wiki Index

### Community 6 - "Document Loading"
Cohesion: 0.70
Nodes (4): load_document(), read_docx(), read_pdf(), read_txt()

### Community 7 - "RAG Eval Dependencies"
Cohesion: 0.67
Nodes (3): HuggingFace Datasets / scikit-learn Evaluation, RAG Retrieval Pipeline, Typer/Rich CLI Framework

## Knowledge Gaps
- **5 isolated node(s):** `Ollama Local LLM Backend`, `Tiktoken Token Counting`, `Graphify Wiki Index`, `HuggingFace Datasets / scikit-learn Evaluation`, `Typer/Rich CLI Framework`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `askQuestionToClaude()` connect `LLM Client & Judge Prompts` to `Evaluation & Diagnosis Pipeline`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `ask_CLAUDE_TO_EVALUATE_RESPONSE()` connect `LLM Client & Judge Prompts` to `Evaluation & Diagnosis Pipeline`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `ask_AI_TO_EVALUTE_RESPONSE()` connect `LLM Client & Judge Prompts` to `Evaluation & Diagnosis Pipeline`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **What connects `Ollama Local LLM Backend`, `Tiktoken Token Counting`, `Graphify Wiki Index` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._