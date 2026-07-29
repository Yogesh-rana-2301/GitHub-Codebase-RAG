import os
import pickle

import faiss
import numpy as np

from chunker import Chunk
from embedder import Embedder


class FAISSRetriever:
    """
    Builds a FAISS vector index from chunks, then retrieves the top-k most
    similar chunks for a given query string.

    Why FAISS?
      FAISS (Facebook AI Similarity Search) runs entirely in-process — no server,
      no Docker, no API key.  For a portfolio project it is the right choice.
      In production you would use Pinecone, Weaviate, or Chroma for persistence
      and scalability, but the retrieval logic is identical.

    Index type: IndexFlatL2
      Exact nearest-neighbour search using L2 (Euclidean) distance on
      L2-normalised vectors, which is equivalent to cosine similarity.
      For large-scale use (millions of vectors) you would switch to
      IndexIVFFlat (approximate) for speed.
    """

    def __init__(self, embedder: Embedder):
        self.embedder: Embedder      = embedder
        self.index:    faiss.Index   = None   # type: ignore
        self.chunks:   list[Chunk]   = []     # parallel array — index i ↔ chunk i

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and load them into the FAISS index."""
        self.chunks = chunks
        texts       = [c.content for c in chunks]

        print(f"  Embedding {len(texts)} chunks ...")
        embeddings  = self.embedder.embed_batch(texts)          # (N, dim)

        # L2-normalise so that L2 distance ≡ cosine distance
        faiss.normalize_L2(embeddings)

        self.index  = faiss.IndexFlatL2(self.embedder.dimension)
        self.index.add(embeddings)
        print(f"  FAISS index ready — {self.index.ntotal} vectors stored.")

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Return the top_k chunks most similar to `query`.
        Each element is (Chunk, distance) — lower distance = more similar.
        """
        if self.index is None:
            raise RuntimeError("Call build() before retrieve().")

        q_vec = self.embedder.embed_one(query).reshape(1, -1)   # (1, dim)
        faiss.normalize_L2(q_vec)

        distances, indices = self.index.search(q_vec, top_k)    # both shape (1, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:   # FAISS returns -1 when there are fewer than top_k vectors
                continue
            results.append((self.chunks[idx], float(dist)))

        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save index and chunk list so we don't have to re-embed every run."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"  Saved retriever to '{directory}'.")

    def load(self, directory: str) -> None:
        """Load a previously saved index."""
        self.index  = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"  Loaded retriever — {len(self.chunks)} chunks from '{directory}'.")
