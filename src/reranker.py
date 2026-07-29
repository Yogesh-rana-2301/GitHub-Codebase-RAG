from sentence_transformers import CrossEncoder

from chunker import Chunk


class Reranker:
    """
    Cross-encoder reranker.

    Why reranking?
      The FAISS retriever uses a *bi-encoder*: query and chunk are each embedded
      independently, then compared with cosine similarity.  This is fast (one
      embedding per query regardless of corpus size) but approximate — the model
      never sees the query and chunk together.

      A *cross-encoder* receives (query, chunk) concatenated as a single input,
      so it can model fine-grained relevance interactions.  This is much more
      accurate but also slower — you only run it on the small set of candidates
      already retrieved by FAISS (typically top-5 or top-10), not the whole corpus.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      Fine-tuned on the MS MARCO passage-ranking dataset.  Small (22 MB),
      fast, and strong at judging query–passage relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker '{model_name}' ...")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        """
        Score each (query, chunk) pair with the cross-encoder, then return
        the top_k chunks sorted by reranker score (higher = more relevant).
        """
        if not candidates:
            return []

        pairs  = [(query, chunk.content) for chunk, _ in candidates]
        scores = self.model.predict(pairs)   # returns list of floats

        # Zip chunks with their new reranker scores and sort descending
        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked[:top_k]
