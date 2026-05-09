"""
scripts/run_phase3_hybrid.py
──────────────────────────────
Phase 3: Hybrid Search & Re-ranking

This script:
  1. Loads text chunks (already embedded from Phase 1)
  2. Builds a BM25 sparse index
  3. Runs hybrid (dense + sparse) retrieval
  4. Applies cross-encoder re-ranking
  5. Tests with sample queries and compares FAISS vs hybrid vs hybrid+rerank

Usage:
    python scripts/run_phase3_hybrid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunk_text     import load_chunks
from src.retrieval.retriever      import Retriever, build_bm25_index
from src.rag_pipeline             import RAGPipeline


TEST_QUERIES = [
    "What was IFC's net income for fiscal year 2024?",
    "How much did IFC commit to climate-related investments?",
    "What is the breakdown of IFC's loan portfolio by region?",
]


def compare_retrievers(query: str, chunks: list[dict]):
    """Compare FAISS, Qdrant, Hybrid and Hybrid+Rerank for one query."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("="*60)

    for mode, rerank in [("faiss", False), ("qdrant", False), ("hybrid", False), ("hybrid", True)]:
        label = f"{mode}" + (" + rerank" if rerank else "")
        try:
            r = Retriever(mode=mode).load(chunks=chunks)
            results = r.retrieve(query, top_k=3, use_rerank=rerank)
            pages = [res["page_number"] for res in results]
            scores = [round(res.get("rerank_score", res.get("score", 0)), 3) for res in results]
            print(f"\n  [{label}]")
            print(f"    Pages retrieved: {pages}")
            print(f"    Scores:          {scores}")
            print(f"    Top chunk preview: {results[0]['text'][:120]}...")
        except Exception as e:
            print(f"  [{label}] Error: {e}")


def main():
    print("=" * 60)
    print("  PHASE 3 – Hybrid Search & Re-ranking")
    print("=" * 60)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} text chunks")

    # Build BM25 index for sparse retrieval
    build_bm25_index(chunks)

    # Compare all retrieval modes on test queries
    for query in TEST_QUERIES:
        compare_retrievers(query, chunks)

    # Final: generate answers with best pipeline (hybrid + rerank)
    print(f"\n{'='*60}")
    print("  Generating answers with Hybrid + Re-ranking pipeline")
    print("="*60)

    pipeline = RAGPipeline(
        retriever_mode="hybrid",
        use_rerank=True,
    ).load(chunks=chunks)

    for query in TEST_QUERIES[:1]:
        print(f"\n❓ {query}")
        print("💡 Answer:")
        for token in pipeline.query(query, stream=True):
            print(token, end="", flush=True)
        print()

    print("\n✅ Phase 3 complete!")


if __name__ == "__main__":
    main()
