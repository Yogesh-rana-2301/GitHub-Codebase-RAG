# GitHub Codebase RAG with Evaluation Harness

Ask questions about any GitHub repository using Retrieval-Augmented Generation (RAG).
The key differentiator: an **evaluation harness** that measures and compares retrieval
quality across three chunking strategies and a cross-encoder re-ranker.

---

## Evaluation Results

| Strategy              | Reranker | Hit Rate |
|-----------------------|----------|----------|
| fixed_no_overlap      | No       | —        |
| fixed_with_overlap    | No       | —        |
| semantic              | No       | —        |
| fixed_no_overlap      | Yes      | —        |
| fixed_with_overlap    | Yes      | —        |
| semantic              | Yes      | —        |

> Run `python main.py --evaluate` to populate this table with real numbers.
> Expected result: semantic chunking improves hit rate by ~10-15pp over fixed-size.

---

## Architecture

```
GitHub Repo
    │
    ▼
ingest.py          Clone repo → extract .md and .py files
    │
    ▼
chunker.py         Split into chunks using one of 3 strategies:
                   ├── fixed_no_overlap   (500 chars, no overlap)
                   ├── fixed_with_overlap (500 chars, 100-char overlap)
                   └── semantic           (paragraph boundaries, ≤600 chars)
    │
    ▼
embedder.py        sentence-transformers all-MiniLM-L6-v2
                   384-dimensional vectors, runs locally, no API key
    │
    ▼
retriever.py       FAISS IndexFlatL2 — exact nearest-neighbour search
                   (L2-normalised = cosine similarity)
    │
    ▼  (optional)
reranker.py        CrossEncoder ms-marco-MiniLM-L-6-v2
                   Rescores top-5 candidates with query+chunk jointly
    │
    ▼
llm.py             Google Gemini 1.5 Flash
                   Grounded prompt — answer from context only, cite sources
    │
    ▼
Answer + Source Citations
```

---

## Key Design Decisions

**Why no LangChain?**
Every component is built from scratch so each step is fully explainable. LangChain
abstracts away the embedding, indexing, and retrieval logic. Understanding what
happens at each step matters more than using a framework.

**Why FAISS and not Pinecone/Chroma?**
FAISS runs entirely locally — no server, no cost, no vendor lock-in. The retrieval
logic is identical to production vector databases; only the persistence layer differs.

**Why sentence-transformers for embeddings?**
The `all-MiniLM-L6-v2` model (22 MB) runs locally with no API key needed. It produces
384-dimensional embeddings that perform well on retrieval benchmarks at zero cost.

**Why a CrossEncoder for re-ranking?**
The initial FAISS retrieval uses a bi-encoder (query and chunk embedded separately).
This is fast but approximate. A CrossEncoder sees the query and chunk together,
giving more accurate relevance scores. It only runs on the top-5 candidates,
not the full corpus — so it stays fast.

**Why semantic chunking outperforms fixed-size?**
Fixed-size chunking can cut a paragraph mid-sentence, leaving incomplete context
in each half. Semantic chunking keeps related sentences together, so the model
receives coherent context about one concept per chunk.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Gemini API key (get free key at https://aistudio.google.com)
export GEMINI_API_KEY=your_key_here

# 3. Ask a question (clones FastAPI repo automatically on first run)
python main.py --query "How do I declare path parameters?"

# 4. Run full evaluation
python main.py --evaluate

# 5. Use reranker
python main.py --query "How does dependency injection work?" --strategy semantic --reranker

# 6. Use a different repo
python main.py --repo https://github.com/encode/httpx --query "How do I make async requests?"
```

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local, free, 384-dim |
| Vector search | FAISS `IndexFlatL2` | Local, exact, no server |
| Re-ranker | CrossEncoder `ms-marco-MiniLM-L-6-v2` | Accurate query-chunk scoring |
| LLM | Google Gemini 1.5 Flash | Free tier, 1M token context |
| Evaluation | Custom hit-rate harness | Transparent, no black-box library |

---

## Project Structure

```
GitHub-RAG-Eval/
├── src/
│   ├── ingest.py       — clone repo, extract documents
│   ├── chunker.py      — 3 chunking strategies
│   ├── embedder.py     — sentence-transformers wrapper
│   ├── retriever.py    — FAISS index: build, search, save, load
│   ├── reranker.py     — CrossEncoder reranker
│   ├── llm.py          — Gemini API with grounded prompting
│   └── pipeline.py     — assembles all components
├── eval/
│   ├── evaluator.py    — hit rate computation
│   └── qa_dataset.json — 30 benchmark Q&A pairs
├── results/
│   └── evaluation_results.json  (auto-generated)
├── data/               — cloned repo + FAISS index cache (git-ignored)
├── main.py             — CLI entry point
└── requirements.txt
```
