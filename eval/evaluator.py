"""
evaluator.py — measures retrieval hit rate across all chunking strategies.

What is "hit rate"?
  For each question in the benchmark dataset, we check whether the FAISS
  retriever returns the correct source file in its top-k results.
  hit_rate = (number of hits) / (total questions) × 100

Why source_file and not exact chunk?
  In codebase RAG, a file may produce many chunks.  If ANY chunk from the
  correct file is in the top-k, the retriever has found the right place.
  Using the file as the unit of correctness is fair and matches how
  interviewers will think about the problem.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from chunker  import STRATEGIES
from embedder import Embedder
from retriever import FAISSRetriever
from reranker  import Reranker


def load_qa_dataset(path: str) -> list[dict]:
    """
    Load the benchmark Q&A dataset.

    Each item must have:
      - "question"    : str
      - "source_file" : str  (relative filepath inside the repo)
      - "answer"      : str  (ground-truth answer, not used for hit-rate)
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retriever(
    documents:       list[dict],
    qa_dataset:      list[dict],
    strategy:        str,
    use_reranker:    bool = False,
    top_k:           int  = 3,
    retrieve_k:      int  = 5,   # candidate pool before reranking
) -> dict:
    """
    Build a retriever for `strategy`, run all questions, compute hit rate.
    Returns a result dict with per-question details and the aggregate hit rate.
    """
    from chunker import get_chunks_by_strategy

    print(f"\n{'─'*55}")
    print(f"  Strategy : {strategy}")
    print(f"  Reranker : {use_reranker}")
    print(f"{'─'*55}")

    # Build embedder + retriever
    embedder  = Embedder()
    retriever = FAISSRetriever(embedder)
    chunks    = get_chunks_by_strategy(documents, strategy)
    retriever.build(chunks)

    reranker = Reranker() if use_reranker else None

    hits    = 0
    details = []

    for item in qa_dataset:
        question     = item["question"]
        correct_file = item["source_file"]

        # Retrieve
        candidates = retriever.retrieve(question, top_k=retrieve_k)

        if use_reranker and reranker:
            final = reranker.rerank(question, candidates, top_k=top_k)
        else:
            final = candidates[:top_k]

        retrieved_files = [chunk.filepath for chunk, _ in final]

        # A "hit" = correct file appears anywhere in retrieved files
        hit = any(correct_file in rf or rf in correct_file for rf in retrieved_files)
        hits += int(hit)

        details.append({
            "question":         question,
            "correct_source":   correct_file,
            "retrieved_sources": retrieved_files,
            "hit":              hit,
        })

    hit_rate = round(hits / len(qa_dataset) * 100, 1)

    print(f"  Hit rate : {hit_rate}%  ({hits}/{len(qa_dataset)} questions)")

    return {
        "strategy":        strategy,
        "use_reranker":    use_reranker,
        "total_questions": len(qa_dataset),
        "hits":            hits,
        "hit_rate":        hit_rate,
        "details":         details,
    }


def run_full_evaluation(
    documents:       list[dict],
    qa_dataset_path: str,
) -> list[dict]:
    """
    Evaluate all 3 strategies × {no reranker, reranker} = 6 combinations.
    Returns a list of result dicts.
    Saves detailed results to results/evaluation_results.json.
    """
    qa_dataset  = load_qa_dataset(qa_dataset_path)
    all_results = []

    for strategy in STRATEGIES:
        for use_reranker in [False, True]:
            result = evaluate_retriever(
                documents    = documents,
                qa_dataset   = qa_dataset,
                strategy     = strategy,
                use_reranker = use_reranker,
            )
            all_results.append(result)

    # Print summary table
    print(f"\n\n{'═'*55}")
    print("  EVALUATION SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'Strategy':<25} {'Reranker':<10} {'Hit Rate'}")
    print(f"  {'─'*25} {'─'*10} {'─'*8}")
    for r in all_results:
        print(f"  {r['strategy']:<25} {str(r['use_reranker']):<10} {r['hit_rate']}%")
    print(f"{'═'*55}\n")

    # Save detailed JSON
    os.makedirs("results", exist_ok=True)
    out_path = "results/evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved → {out_path}")

    return all_results
