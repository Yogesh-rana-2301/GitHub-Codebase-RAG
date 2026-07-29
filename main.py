"""
main.py — command-line interface for the GitHub RAG Eval project.

Usage examples:

  # Index FastAPI repo and ask one question
  python main.py --repo https://github.com/tiangolo/fastapi \\
                 --query "How do I declare path parameters?"

  # Use semantic chunking + reranker
  python main.py --query "How does dependency injection work?" \\
                 --strategy semantic --reranker

  # Run the full evaluation across all 6 strategy combinations
  python main.py --evaluate

  # Skip cloning if already done
  python main.py --query "How do I handle CORS?"
"""

import argparse
import json
import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

from ingest     import clone_repo, extract_documents
from pipeline   import RAGPipeline
from chunker    import STRATEGIES
from evaluator  import run_full_evaluation

DEFAULT_REPO = "https://github.com/tiangolo/fastapi"
REPO_DIR     = "data/repo"
INDEX_DIR    = "data/index_cache"
QA_DATASET   = "eval/qa_dataset.json"


def main():
    parser = argparse.ArgumentParser(
        description="GitHub Codebase RAG with Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo", type=str, default=DEFAULT_REPO,
        help=f"GitHub repo URL to index (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Ask a question about the indexed repository"
    )
    parser.add_argument(
        "--strategy", type=str, default="semantic", choices=STRATEGIES,
        help="Chunking strategy to use for querying (default: semantic)"
    )
    parser.add_argument(
        "--reranker", action="store_true",
        help="Use cross-encoder reranker on retrieved candidates"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run full evaluation across all chunking strategies and print results table"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")

    # ── Step 1: Ingest the repository ────────────────────────────────────────
    repo_dir  = clone_repo(args.repo, REPO_DIR)
    documents = extract_documents(repo_dir)

    # ── Step 2a: Full evaluation mode ────────────────────────────────────────
    if args.evaluate:
        print("\nStarting full evaluation across all strategies ...")
        results = run_full_evaluation(documents, QA_DATASET)

        # Pretty-print the final table
        print("\n  Copy these numbers into your README:\n")
        print(f"  {'Strategy':<25} {'Reranker':<10} {'Hit Rate'}")
        print(f"  {'─'*25} {'─'*10} {'─'*8}")
        for r in results:
            print(f"  {r['strategy']:<25} {str(r['use_reranker']):<10} {r['hit_rate']}%")
        return

    # ── Step 2b: Single-query mode ───────────────────────────────────────────
    if args.query:
        pipeline = RAGPipeline(
            strategy     = args.strategy,
            use_reranker = args.reranker,
            api_key      = api_key,
        )
        pipeline.build(documents, cache_dir=INDEX_DIR)

        result = pipeline.query(args.query)

        print(f"\n{'═'*60}")
        print(f"  Q: {result['question']}")
        print(f"{'─'*60}")
        print(f"  A: {result['answer']}")
        print(f"{'─'*60}")
        print(f"  Sources : {', '.join(result['source_files'])}")
        print(f"  Strategy: {result['strategy']}  |  Reranked: {result['reranked']}")
        print(f"{'═'*60}\n")
        return

    # ── No action specified ──────────────────────────────────────────────────
    parser.print_help()


if __name__ == "__main__":
    main()
