import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps a sentence-transformers model to convert text → dense vector.

    Why sentence-transformers and not OpenAI embeddings?
      - Runs entirely locally — no API key, no cost, no network latency for embedding.
      - `all-MiniLM-L6-v2` is 22 MB, fast, and produces 384-dimensional vectors
        that perform very well on retrieval benchmarks.
      - In interviews: "I chose local embeddings to keep the pipeline self-contained
        and avoid dependency on an external paid API for a non-generative step."
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model '{model_name}' ...")
        self.model     = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dimension}")

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns a float32 numpy array of shape (len(texts), self.dimension).
        show_progress_bar=True is useful when embedding thousands of chunks.
        """
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dimension,)."""
        return self.model.encode(
            [text],
            convert_to_numpy=True,
        ).astype("float32")[0]
