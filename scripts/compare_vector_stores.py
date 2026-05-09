"""
scripts/compare_vector_stores.py
──────────────────────────────────
Standalone script to compare FAISS vs Qdrant on:
  1. Search speed (milliseconds per query)
  2. Result quality (do they return the same pages?)
  3. Memory usage
  4. Metadata filtering capability (Qdrant only)

Usage:
    python scripts/compare_vector_stores.py
"""

import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.embeddings.embed        import load_embeddings
from src.embeddings.faiss_store  import build_faiss_index, load_faiss_index, search_faiss
from src.embeddings.qdrant_store import build_qdrant_collection, get_qdrant_client, search_qdrant
from src.retrieval.retriever     import embed_query
from src.ingestion.chunk_text    import load_chunks


TEST_QUERIES = [
    "What was IFC's net income for fiscal year 2024?",
    "What is the total value of IFC's assets?",
    "How much did IFC commit to climate investments?",
    "What are the main risk factors?",
    "Describe IFC's loan portfolio by region.",
]

SEPARATOR = "─" * 60


def measure_faiss(queries: list[str], index, meta, top_k=5) -> dict:
    """Benchmark FAISS search."""
    times = []
    all_results = []

    for q in queries:
        q_emb = embed_query(q)

        t0 = time.perf_counter()
        results = search_faiss(q_emb, index, meta, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        times.append(elapsed_ms)
        all_results.append([r["page_number"] for r in results])

    return {
        "avg_ms"   : np.mean(times),
        "min_ms"   : np.min(times),
        "max_ms"   : np.max(times),
        "pages"    : all_results,
    }


def measure_qdrant(queries: list[str], client, top_k=5) -> dict:
    """Benchmark Qdrant search."""
    times = []
    all_results = []

    for q in queries:
        q_emb = embed_query(q)

        t0 = time.perf_counter()
        results = search_qdrant(q_emb, client, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        times.append(elapsed_ms)
        all_results.append([r["page_number"] for r in results])

    return {
        "avg_ms"   : np.mean(times),
        "min_ms"   : np.min(times),
        "max_ms"   : np.max(times),
        "pages"    : all_results,
    }


def result_overlap(pages_a: list, pages_b: list) -> float:
    """What fraction of top-5 results overlap between two stores?"""
    overlaps = []
    for a, b in zip(pages_a, pages_b):
        set_a, set_b = set(a), set(b)
        if not set_a and not set_b:
            overlaps.append(1.0)
        elif not set_a or not set_b:
            overlaps.append(0.0)
        else:
            overlaps.append(len(set_a & set_b) / len(set_a | set_b))
    return float(np.mean(overlaps))


def main():
    print(SEPARATOR)
    print("  FAISS vs Qdrant – Vector Store Comparison")
    print(SEPARATOR)

    # Load data
    chunks   = load_chunks()
    embedded = load_embeddings()
    print(f"\nLoaded {len(embedded)} embeddings  |  dim={len(embedded[0]['embedding'])}")

    # ── Build / load FAISS ────────────────────────────────────────────────
    print("\n📦 Loading FAISS index...")
    try:
        faiss_index, faiss_meta = load_faiss_index()
    except Exception:
        print("  Index not found – building...")
        faiss_index, faiss_meta = build_faiss_index(embedded)

    # ── Build / connect Qdrant ────────────────────────────────────────────
    print("\n📦 Connecting to Qdrant...")
    qdrant_available = False
    try:
        qdrant_client = get_qdrant_client()
        qdrant_available = True
    except Exception as e:
        print(f"  ⚠️  Qdrant not available: {e}")
        print("  Start Qdrant with: cd docker && docker-compose up -d")

    # ── Speed benchmark ───────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Speed Benchmark")
    print(SEPARATOR)

    print("\nEmbedding queries (same cost for both stores)...")
    # Pre-embed to isolate store performance
    _ = [embed_query(q) for q in TEST_QUERIES]   # warm up

    print("\nBenchmarking FAISS...")
    faiss_bench = measure_faiss(TEST_QUERIES, faiss_index, faiss_meta)

    if qdrant_available:
        print("Benchmarking Qdrant...")
        qdrant_bench = measure_qdrant(TEST_QUERIES, qdrant_client)

    print(f"\n{'Store':<12} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 45)
    print(f"{'FAISS':<12} {faiss_bench['avg_ms']:>10.2f} {faiss_bench['min_ms']:>10.2f} {faiss_bench['max_ms']:>10.2f}")
    if qdrant_available:
        print(f"{'Qdrant':<12} {qdrant_bench['avg_ms']:>10.2f} {qdrant_bench['min_ms']:>10.2f} {qdrant_bench['max_ms']:>10.2f}")

    # ── Result quality overlap ────────────────────────────────────────────
    if qdrant_available:
        overlap = result_overlap(faiss_bench["pages"], qdrant_bench["pages"])
        print(f"\n  Result overlap (Jaccard): {overlap:.1%}")
        print("  (High overlap = both stores find the same relevant chunks)")

    # ── Per-query breakdown ───────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Per-Query Results")
    print(SEPARATOR)

    for i, q in enumerate(TEST_QUERIES):
        print(f"\nQuery {i+1}: {q[:65]}...")
        print(f"  FAISS  pages: {faiss_bench['pages'][i]}")
        if qdrant_available:
            print(f"  Qdrant pages: {qdrant_bench['pages'][i]}")

    # ── Memory usage comparison ───────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Memory Usage")
    print(SEPARATOR)

    tracemalloc.start()
    import pickle
    from config.settings import FAISS_INDEX_PATH
    with open(FAISS_INDEX_PATH + ".pkl", "rb") as f:
        _ = pickle.load(f)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  FAISS metadata in RAM: {peak / 1024 / 1024:.1f} MB")
    print("  Qdrant data on disk (Docker volume): persistent, not in RAM")

    # ── Feature comparison table ──────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Feature Comparison Summary")
    print(SEPARATOR)
    print("""
  Feature                         FAISS           Qdrant
  ─────────────────────────────── ─────────────── ───────────────
  Search speed (small corpus)     ✅ Very fast    ⚡ Fast (HTTP)
  Metadata filtering              ❌ Manual       ✅ Built-in
  Persistence across restarts     ❌ File-based   ✅ Docker volume
  Horizontal scaling              ❌ Single node  ✅ Distributed
  REST API                        ❌ Python only  ✅ Full API
  Real-time updates               ⚠️  Rebuild      ✅ Upsert
  Best for                        Prototyping     Production
    """)

    print("✅ Comparison complete!")


if __name__ == "__main__":
    main()
