"""
pipeline.py — assembles ingest → chunk → embed → retrieve → (rerank) → generate
into a single RAGPipeline object.

Interview explanation (one sentence):
  "The pipeline takes a chunking strategy and optional reranker flag, builds a
   FAISS index over the repo, then for each query retrieves the top candidates,
   optionally reranks them with a cross-encoder, and sends the final context to
   Gemini with a grounded prompt that forces source citation."
"""

import os
import sys

# Allow running src/ files directly
sys.path.insert(0, os.path.dirname(__file__))

from chunker  import get_chunks_by_strategy, Chunk, STRATEGIES
from embedder import Embedder
from retriever import FAISSRetriever
from reranker  import Reranker
from llm       import GeminiLLM


class RAGPipeline:
    """
    Full RAG pipeline.

    Parameters
    ----------
    strategy     : one of "fixed_no_overlap", "fixed_with_overlap", "semantic"
    use_reranker : if True, retrieved candidates are re-scored with a CrossEncoder
    api_key      : Gemini API key (falls back to GEMINI_API_KEY env var)
    """

    def __init__(
        self,
        strategy:     str  = "semantic",
        use_reranker: bool = False,
        api_key:      str  = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")

        self.strategy     = strategy
        self.use_reranker = use_reranker

        self.embedder  = Embedder()
        self.retriever = FAISSRetriever(self.embedder)
        self.reranker  = Reranker() if use_reranker else None
        self.llm       = GeminiLLM(api_key=api_key)

        self._built = False

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, documents: list[dict], cache_dir: str = None) -> int:
        """
        Chunk the documents with the chosen strategy and build the FAISS index.

        If cache_dir is given, the index is saved so it can be reloaded on the
        next run without re-embedding (saves ~30 seconds for large repos).

        Returns the total number of chunks created.
        """
        # Try loading from cache first
        if cache_dir:
            index_path = os.path.join(cache_dir, self.strategy)
            if os.path.exists(index_path):
                print(f"Loading cached index for strategy '{self.strategy}' ...")
                self.retriever.load(index_path)
                self._built = True
                return len(self.retriever.chunks)

        print(f"\nBuilding pipeline — strategy: '{self.strategy}' ...")
        chunks = get_chunks_by_strategy(documents, self.strategy)
        print(f"  Created {len(chunks)} chunks.")

        self.retriever.build(chunks)

        if cache_dir:
            self.retriever.save(os.path.join(cache_dir, self.strategy))

        self._built = True
        return len(chunks)

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str, retrieve_k: int = 5, final_k: int = 3) -> dict:
        """
        Run a full RAG query.

        1. Retrieve top `retrieve_k` chunks from FAISS.
        2. If use_reranker: re-score with CrossEncoder, keep top `final_k`.
        3. Send the final chunks + question to Gemini.
        4. Return a dict with the answer, sources, and metadata.

        retrieve_k > final_k because the reranker needs a larger candidate pool
        to be useful — it selects the best `final_k` from `retrieve_k` options.
        """
        if not self._built:
            raise RuntimeError("Call build() before query().")

        # Step 1 — Retrieve
        candidates: list[tuple[Chunk, float]] = self.retriever.retrieve(
            question, top_k=retrieve_k
        )

        # Step 2 — Rerank (optional)
        if self.use_reranker and self.reranker:
            final = self.reranker.rerank(question, candidates, top_k=final_k)
        else:
            final = candidates[:final_k]

        # Step 3 — Generate
        context_texts = [chunk.content  for chunk, _ in final]
        source_files  = [chunk.filepath for chunk, _ in final]

        answer = self.llm.answer(question, context_texts, source_files)

        return {
            "question":        question,
            "answer":          answer,
            "source_files":    source_files,
            "chunk_ids":       [chunk.chunk_id for chunk, _ in final],
            "strategy":        self.strategy,
            "reranked":        self.use_reranker,
        }
