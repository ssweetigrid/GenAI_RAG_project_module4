"""
scripts/run_phase4_advanced.py
───────────────────────────────
Phase 4: Advanced RAG Techniques

This script demonstrates and tests both Phase 4 features:

  Task 1 – Semantic Caching
    • Shows cache miss on first call (hits Gemini)
    • Shows cache hit on second similar call (instant, no API call)
    • Demonstrates TTL (time-to-live) expiry
    • Warms cache with common questions
    • Prints cache statistics and exports to CSV

  Task 2 – Multi-hop Retrieval
    • Auto-detects whether a question needs multi-hop
    • Decomposes complex questions into sub-questions via Gemini
    • Runs iterative retrieval hops with context carry-forward
    • Synthesises a final answer from all partial answers
    • Compares single-hop vs multi-hop answer quality

Usage:
    python scripts/run_phase4_advanced.py
    python scripts/run_phase4_advanced.py --task cache       # only Task 1
    python scripts/run_phase4_advanced.py --task multihop    # only Task 2
    python scripts/run_phase4_advanced.py --task both        # default
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline               import RAGPipeline
from src.ingestion.chunk_text       import load_chunks
from src.retrieval.semantic_cache   import SemanticCache, cached_rag_query
from src.retrieval.multihop_retriever import (
    MultiHopRetriever, needs_multihop, decompose_question
)

SEPARATOR = "═" * 62


# ─────────────────────────────────────────────────────────────────────────
#  TASK 1 – SEMANTIC CACHING
# ─────────────────────────────────────────────────────────────────────────

# Questions to demonstrate cache behaviour
# Each group is semantically similar – same topic, different phrasing
CACHE_DEMO_GROUPS = [
    {
        "label": "Net Income",
        "queries": [
            "What was IFC's net income for fiscal year 2024?",          # miss
            "How much net income did IFC earn in 2024?",                 # hit
            "Tell me IFC's profit for the 2024 financial year.",         # hit
        ],
    },
    {
        "label": "Total Assets",
        "queries": [
            "What is the total value of IFC's assets as of June 2024?", # miss
            "What are IFC's total assets in 2024?",                      # hit
        ],
    },
    {
        "label": "Climate Investment",
        "queries": [
            "How much did IFC commit to climate-related investments?",   # miss
            "What were IFC's commitments to green or climate finance?",  # hit (maybe)
        ],
    },
]

# Questions to warm the cache at startup
WARMUP_QUESTIONS = [
    "What was IFC's net income for fiscal year 2024?",
    "What are the main risk factors in the IFC annual report?",
    "What was IFC's return on equity?",
]


def run_task1_semantic_caching(pipeline):
    """Demonstrate and test the semantic caching layer."""

    print(f"\n{SEPARATOR}")
    print("  TASK 1 – Semantic Caching")
    print(SEPARATOR)

    # ── 1a. Basic demo: miss → hit ────────────────────────────────────────
    print("\n📦 1a. Cache miss then hit demo")
    print("─" * 45)

    cache = SemanticCache(threshold=0.92)
    cache.clear()   # Start fresh for this demo

    group = CACHE_DEMO_GROUPS[0]  # Use the "Net Income" group
    timings = []

    for i, query in enumerate(group["queries"]):
        print(f"\nQuery {i+1}: '{query}'")
        t0 = time.perf_counter()
        answer, from_cache = cached_rag_query(query, pipeline, cache, verbose=True)
        elapsed = time.perf_counter() - t0

        label = "⚡ CACHE HIT" if from_cache else "🌐 API CALL"
        print(f"  {label}  |  time={elapsed:.2f}s")
        print(f"  Answer preview: {answer[:100]}...")
        timings.append((query[:50], from_cache, elapsed))

    # Print timing summary
    print("\n  ⏱️  Timing Summary:")
    print(f"  {'Query':<52} {'Source':<12} {'Time'}")
    print("  " + "-" * 72)
    for q, from_cache, t in timings:
        source = "cache" if from_cache else "Gemini API"
        print(f"  {q:<52} {source:<12} {t:.2f}s")

    # ── 1b. All groups ────────────────────────────────────────────────────
    print(f"\n\n📦 1b. Full group demo ({len(CACHE_DEMO_GROUPS)} topics)")
    print("─" * 45)

    cache.clear()

    for group in CACHE_DEMO_GROUPS:
        print(f"\n  Topic: {group['label']}")
        for i, query in enumerate(group["queries"]):
            answer, from_cache = cached_rag_query(query, pipeline, cache, verbose=True)
            status = "HIT ✅" if from_cache else "MISS ❌"
            print(f"    {i+1}. [{status}] {query[:60]}...")

    # ── 1c. Cache statistics ──────────────────────────────────────────────
    cache.print_stats()

    # ── 1d. Cache warming ─────────────────────────────────────────────────
    print(f"\n📦 1c. Cache warming")
    print("─" * 45)
    print("Pre-populating cache with common questions...")
    fresh_cache = SemanticCache(threshold=0.92)
    fresh_cache.warm(WARMUP_QUESTIONS, pipeline)
    fresh_cache.print_stats()

    # ── 1e. TTL demonstration ─────────────────────────────────────────────
    print(f"\n📦 1d. TTL (Time-to-Live) demonstration")
    print("─" * 45)
    print("Creating a cache with 3-second TTL...")

    ttl_cache = SemanticCache(threshold=0.92, ttl_seconds=3)
    test_q = "What was IFC's net income for fiscal year 2024?"

    answer, from_cache = cached_rag_query(test_q, pipeline, ttl_cache, verbose=False)
    print(f"  First call:   {'cache' if from_cache else 'API'}")

    answer, from_cache = cached_rag_query(test_q, pipeline, ttl_cache, verbose=False)
    print(f"  Second call:  {'cache' if from_cache else 'API'} (immediate)")

    print("  Waiting 4 seconds for TTL to expire...")
    time.sleep(4)

    answer, from_cache = cached_rag_query(test_q, pipeline, ttl_cache, verbose=False)
    print(f"  After TTL:    {'cache' if from_cache else 'API'} (should be API again)")

    # ── 1f. Export CSV ────────────────────────────────────────────────────
    print(f"\n📦 1e. Exporting cache to CSV")
    export_path = cache.export_csv()
    print(f"  Inspect at: {export_path}")

    print(f"\n✅ Task 1 (Semantic Caching) complete!")


# ─────────────────────────────────────────────────────────────────────────
#  TASK 2 – MULTI-HOP RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────

# Simple questions (should be single-hop)
SIMPLE_QUESTIONS = [
    "What was IFC's net income in 2024?",
    "What is IFC's total assets figure?",
]

# Complex questions (should trigger multi-hop)
COMPLEX_QUESTIONS = [
    (
        "How did IFC's net income change between fiscal year 2023 and 2024, "
        "and what were the main risk factors that contributed to this change?",
        3,  # max_hops
    ),
    (
        "What was IFC's equity investment strategy and how did it perform "
        "compared to its loan portfolio in terms of income generation?",
        2,
    ),
    (
        "Describe the relationship between IFC's climate finance commitments "
        "and its overall financial performance in 2024.",
        3,
    ),
]


def run_task2_multihop(pipeline):
    """Demonstrate multi-hop retrieval on complex questions."""

    print(f"\n{SEPARATOR}")
    print("  TASK 2 – Multi-hop Retrieval")
    print(SEPARATOR)

    retriever = MultiHopRetriever(pipeline, max_hops=3, top_k=4)

    # ── 2a. needs_multihop() detector ────────────────────────────────────
    print("\n🔎 2a. Multi-hop necessity detector")
    print("─" * 45)

    all_questions = [(q, False) for q in SIMPLE_QUESTIONS] + \
                    [(q, True)  for q, _ in COMPLEX_QUESTIONS]

    print(f"  {'Question':<65} {'Expected':<10} {'Detected'}")
    print("  " + "-" * 85)
    for question, expected_multihop in all_questions:
        detected = needs_multihop(question)
        match = "✅" if detected == expected_multihop else "⚠️ "
        exp_label = "multi" if expected_multihop else "single"
        det_label = "multi" if detected else "single"
        print(f"  {match} {question[:63]:<65} {exp_label:<10} {det_label}")

    # ── 2b. Decomposition demo ────────────────────────────────────────────
    print(f"\n\n🔎 2b. Question decomposition")
    print("─" * 45)

    for question, max_hops in COMPLEX_QUESTIONS[:2]:
        print(f"\n  Complex question:")
        print(f"  '{question[:80]}...'")
        print(f"\n  Gemini decomposes into:")
        sub_questions = decompose_question(question, max_hops=max_hops)
        for i, sq in enumerate(sub_questions, 1):
            print(f"    {i}. {sq}")

    # ── 2c. Full multi-hop run ────────────────────────────────────────────
    print(f"\n\n🔗 2c. Full multi-hop retrieval")
    print("─" * 45)

    question, max_hops = COMPLEX_QUESTIONS[0]
    print(f"\n  Question: '{question[:80]}...'")
    print(f"  Max hops: {max_hops}")

    result = retriever.run(question)

    # Print all hops
    print(f"\n  {'─'*55}")
    print(f"  Hops executed: {result.total_hops}")
    for hop in result.hops:
        print(f"\n  ── Hop {hop.hop_number} ──")
        print(f"  Sub-question: {hop.sub_question}")
        print(f"  Pages retrieved: {[c['page_number'] for c in hop.retrieved_chunks]}")
        print(f"  Partial answer: {hop.partial_answer[:200]}...")

    print(f"\n  {'─'*55}")
    print(f"  FINAL SYNTHESISED ANSWER:")
    print(f"  {'─'*55}")
    print(f"  {result.final_answer[:600]}...")

    # ── 2d. Compare single-hop vs multi-hop ──────────────────────────────
    print(f"\n\n📊 2d. Single-hop vs Multi-hop comparison")
    print("─" * 45)

    test_question = COMPLEX_QUESTIONS[0][0]
    print(f"\n  Question: '{test_question[:80]}...'")

    # Single-hop answer
    print("\n  [Single-hop] Running...")
    t0 = time.perf_counter()
    single_answer = retriever.run_single_hop(test_question)
    single_time   = time.perf_counter() - t0

    # Multi-hop answer
    print("\n  [Multi-hop] Running...")
    t0 = time.perf_counter()
    multi_result = retriever.run(test_question)
    multi_time   = time.perf_counter() - t0

    print(f"\n  ┌{'─'*55}")
    print(f"  │ SINGLE-HOP ({single_time:.1f}s)")
    print(f"  ├{'─'*55}")
    print(f"  │ {single_answer[:300].replace(chr(10), chr(10) + '  │ ')}...")
    print(f"  └{'─'*55}")
    print()
    print(f"  ┌{'─'*55}")
    print(f"  │ MULTI-HOP  ({multi_time:.1f}s, {multi_result.total_hops} hops)")
    print(f"  ├{'─'*55}")
    print(f"  │ {multi_result.final_answer[:400].replace(chr(10), chr(10) + '  │ ')}...")
    print(f"  └{'─'*55}")

    print(f"\n  Key differences:")
    print(f"    • Single-hop retrieved from {TOP_K} chunks in 1 search")
    print(f"    • Multi-hop ran {multi_result.total_hops} targeted searches "
          f"and combined the results")
    print(f"    • Multi-hop is better for questions requiring info from "
          f"multiple sections")
    print(f"    • Single-hop is faster ({single_time:.1f}s vs {multi_time:.1f}s)")

    print(f"\n✅ Task 2 (Multi-hop Retrieval) complete!")


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Advanced RAG Techniques")
    parser.add_argument(
        "--task",
        default="both",
        choices=["cache", "multihop", "both"],
        help="Which task to run: cache | multihop | both",
    )
    args = parser.parse_args()

    print(SEPARATOR)
    print("  PHASE 4 – Advanced RAG Techniques")
    print(SEPARATOR)

    # Load pipeline (required by both tasks)
    print("\n⚙️  Loading RAG pipeline (FAISS mode)...")
    chunks   = load_chunks()
    pipeline = RAGPipeline(retriever_mode="faiss").load(chunks=chunks)
    print("✅ Pipeline ready\n")

    if args.task in ("cache", "both"):
        run_task1_semantic_caching(pipeline)

    if args.task in ("multihop", "both"):
        run_task2_multihop(pipeline)

    print(f"\n{SEPARATOR}")
    print("  Phase 4 Complete! ✅")
    print(SEPARATOR)
    print("\nFiles produced:")
    print("  data/processed/semantic_cache.json     – cached query/answers")
    print("  data/processed/semantic_cache_export.csv – human-readable cache dump")


if __name__ == "__main__":
    from config.settings import TOP_K
    main()
