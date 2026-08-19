# Hybrid RAG Pipeline

A RAG chatbot that answers FastAPI questions using its own documentation as the knowledge base.

## How it works

Docs are fetched from the FastAPI repo, filtered down to real documentation content, split into chunks by markdown header, and embedded with OpenAI's `text-embedding-3-small` into a persistent local Chroma vector store.

At query time, retrieval runs two ways in parallel: BM25 keyword search and vector similarity search. The two ranked lists are combined with Reciprocal Rank Fusion, then re-scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) for a final precision pass. The top results go to `gpt-4o-mini`, which generates a grounded, cited answer. Prompted to admit when the docs don't cover something rather than guess.

Every stage of every query (retrieval, fusion, reranking, generation) is traced end-to-end with Langfuse.

## Evaluation

Quality is measured against a 17-question golden dataset spanning exact-term lookups, conceptual/comparative questions, deliberate out-of-scope questions (testing refusal behavior), and edge cases. Each question is scored two ways: a retrieval hit-rate check (did the system find the right source document) and an LLM-as-judge pass grading answer quality against a rubric.

Current results: 14/14 retrieval hits, 16/17 on judged answer quality.

## Tech Stack

Python · OpenAI (embeddings + generation) · ChromaDB (vector store) · rank_bm25 (keyword search) · sentence-transformers (cross-encoder reranking) · Langfuse (observability/tracing) · uv (dependency management)

## How To Run

```
uv sync
# add OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY to .env

uv run python scripts/fetch_fastapi_docs.py
uv run python -m ingestion.build_corpus
uv run python ingestion/embed.py
uv run python -m ingestion.vector_store

uv run python main.py
uv run python -m evaluation.run_eval
```
