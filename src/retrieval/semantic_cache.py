"""
src/retrieval/semantic_cache.py
────────────────────────────────
Phase 4, Task 1 – Semantic Caching

How it works:
  ┌─────────────────────────────────────────────────────────────┐
  │  User asks a question                                       │
  │       ↓                                                     │
  │  Embed the question → 768-dim vector                        │
  │       ↓                                                     │
  │  Compare with ALL cached question vectors (cosine sim)      │
  │       ↓                              ↓                      │
  │  similarity >= threshold?      similarity < threshold?      │
  │  CACHE HIT → return stored     CACHE MISS → call pipeline   │
  │  answer instantly (no API)          ↓                       │
  │                              Generate answer with Gemini    │
  │                                     ↓                       │
  │                              Store (query, embedding,       │
  │                              answer) in cache for next time │
  └─────────────────────────────────────────────────────────────┘

Benefits:
  • Saves Gemini API calls (cost + latency)
  • Identical or rephrased questions reuse cached answers
  • Cache survives restarts (stored to disk as JSON)

Extra features in this implementation:
  • TTL (time-to-live): entries expire after N seconds
  • Similarity score returned alongside the cached answer
  • Cache statistics: hits, misses, hit-rate
  • Cache warming: pre-populate from a list of known questions
  • Export cache to CSV for inspection
"""

import json
import time
import csv
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import ROOT_DIR
from src.retrieval.retriever import embed_query


# ── Constants ─────────────────────────────────────────────────────────────
CACHE_FILE           = ROOT_DIR / "data" / "processed" / "semantic_cache.json"
DEFAULT_THRESHOLD    = 0.92      # cosine similarity needed for a cache hit
DEFAULT_TTL_SECONDS  = None      # None = entries never expire


class SemanticCache:
    """
    In-memory + disk-backed semantic cache for RAG answers.

    Parameters
    ----------
    threshold   : float   cosine similarity above which a query is a "hit"
    ttl_seconds : int     seconds before an entry expires (None = never)
    cache_file  : Path    where to persist the cache on disk
    """

    def __init__(
        self,
        threshold  : float         = DEFAULT_THRESHOLD,
        ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
        cache_file : Path          = CACHE_FILE,
    ):
        self.threshold   = threshold
        self.ttl_seconds = ttl_seconds
        self.cache_file  = Path(cache_file)

        # Runtime counters – reset every run, not persisted
        self._hits   = 0
        self._misses = 0

        # The actual cache entries (loaded from disk)
        self.entries: list[dict] = []
        self._load_from_disk()

    # ── Disk persistence ──────────────────────────────────────────────────

    def _load_from_disk(self):
        """Load existing cache from disk on startup."""
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            # Prune any already-expired entries on load
            self._prune_expired()
            print(f"📦 Semantic cache loaded  |  {len(self.entries)} entries  "
                  f"|  threshold={self.threshold}")
        else:
            print(f"📦 Semantic cache empty   |  threshold={self.threshold}")

    def _save_to_disk(self):
        """Persist the current cache entries to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False)

    # ── TTL management ────────────────────────────────────────────────────

    def _is_expired(self, entry: dict) -> bool:
        """Return True if this entry has passed its TTL."""
        if self.ttl_seconds is None:
            return False
        age = time.time() - entry.get("timestamp", 0)
        return age > self.ttl_seconds

    def _prune_expired(self):
        """Remove all expired entries from memory (and save)."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if not self._is_expired(e)]
        removed = before - len(self.entries)
        if removed:
            print(f"⏰ Pruned {removed} expired cache entries")
            self._save_to_disk()

    # ── Core similarity ───────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Cosine similarity between two vectors (returns 0.0–1.0)."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _find_best_match(self, q_embedding: list) -> tuple[int, float]:
        """
        Scan all cache entries and return (index, similarity) of the
        best match. Returns (-1, 0.0) if the cache is empty.
        """
        best_sim = 0.0
        best_idx = -1
        for i, entry in enumerate(self.entries):
            if self._is_expired(entry):
                continue
            sim = self._cosine_similarity(q_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, best_sim

    # ── Public API ────────────────────────────────────────────────────────

    def lookup(self, query: str) -> tuple[Optional[str], float]:
        """
        Check if a semantically similar query is already cached.

        Returns
        -------
        (answer, similarity)  if cache hit  → similarity >= threshold
        (None,   similarity)  if cache miss → similarity < threshold
        """
        if not self.entries:
            self._misses += 1
            return None, 0.0

        q_emb = embed_query(query)
        best_idx, best_sim = self._find_best_match(q_emb)

        if best_idx >= 0 and best_sim >= self.threshold:
            # ── Cache HIT ────────────────────────────────────────────────
            self._hits += 1
            self.entries[best_idx]["hits"] += 1
            self.entries[best_idx]["last_accessed"] = time.time()
            self._save_to_disk()

            matched_query = self.entries[best_idx]["query"]
            print(f"  ✅ Cache HIT   similarity={best_sim:.4f}  "
                  f"matched='{matched_query[:55]}...'")
            return self.entries[best_idx]["answer"], best_sim

        # ── Cache MISS ───────────────────────────────────────────────────
        self._misses += 1
        print(f"  ❌ Cache MISS  best_similarity={best_sim:.4f}")
        return None, best_sim

    def store(self, query: str, answer: str, context_chunks: list[dict] = None):
        """
        Save a new query-answer pair to the cache.

        Parameters
        ----------
        query          : the user's question
        answer         : the generated answer
        context_chunks : (optional) which chunks were used – stored as metadata
        """
        q_emb = embed_query(query)

        entry = {
            "query"        : query,
            "embedding"    : q_emb,
            "answer"       : answer,
            "timestamp"    : time.time(),
            "last_accessed": time.time(),
            "hits"         : 0,
            # Store minimal context metadata (not full text, to keep file small)
            "source_pages" : [c.get("page_number") for c in (context_chunks or [])],
        }
        self.entries.append(entry)
        self._save_to_disk()
        print(f"  💾 Cached   '{query[:60]}...'  ({len(self.entries)} entries total)")

    def stats(self) -> dict:
        """Return cache performance statistics for this session."""
        total_calls = self._hits + self._misses
        hit_rate    = self._hits / total_calls if total_calls > 0 else 0.0
        total_hits  = sum(e.get("hits", 0) for e in self.entries)  # all-time

        return {
            "entries_on_disk"  : len(self.entries),
            "session_hits"     : self._hits,
            "session_misses"   : self._misses,
            "session_hit_rate" : f"{hit_rate:.1%}",
            "alltime_reuses"   : total_hits,
            "threshold"        : self.threshold,
            "ttl_seconds"      : self.ttl_seconds,
        }

    def print_stats(self):
        """Pretty-print the cache statistics."""
        s = self.stats()
        print("\n" + "─" * 45)
        print("  📊 Semantic Cache Statistics")
        print("─" * 45)
        for key, val in s.items():
            print(f"  {key:<22} {val}")
        print("─" * 45)

    def warm(self, queries: list[str], pipeline):
        """
        Pre-populate the cache with answers for a list of known questions.
        Call this once at startup to make frequent queries instant.

        Parameters
        ----------
        queries  : list of question strings to pre-answer
        pipeline : RAGPipeline instance to generate answers
        """
        print(f"\n🔥 Warming cache with {len(queries)} queries...")
        for q in queries:
            # Skip if already cached
            cached_answer, sim = self.lookup(q)
            if cached_answer is not None:
                print(f"  ⏭️  Already cached: '{q[:55]}...'")
                continue

            print(f"  Generating: '{q[:55]}...'")
            chunks = pipeline.get_context(q)
            answer = pipeline.query(q)
            self.store(q, answer, context_chunks=chunks)

        print(f"✅ Cache warming complete  |  {len(self.entries)} total entries")

    def export_csv(self, out_path: Path = None) -> Path:
        """
        Export all cache entries to a CSV file for inspection.
        Useful for debugging: see what questions are cached and how often they're hit.
        """
        if out_path is None:
            out_path = CACHE_FILE.parent / "semantic_cache_export.csv"

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["query", "answer_preview", "hits", "source_pages",
                            "timestamp", "last_accessed"],
            )
            writer.writeheader()
            for e in self.entries:
                writer.writerow({
                    "query"         : e["query"],
                    "answer_preview": e["answer"][:120].replace("\n", " "),
                    "hits"          : e.get("hits", 0),
                    "source_pages"  : str(e.get("source_pages", [])),
                    "timestamp"     : time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(e.get("timestamp", 0))
                    ),
                    "last_accessed" : time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(e.get("last_accessed", 0))
                    ),
                })
        print(f"💾 Cache exported → {out_path}")
        return out_path

    def clear(self):
        """Delete all cache entries from memory and disk."""
        count = len(self.entries)
        self.entries = []
        self._save_to_disk()
        print(f"🗑️  Cache cleared  ({count} entries removed)")


# ── Convenience wrapper ───────────────────────────────────────────────────

def cached_rag_query(
    query   : str,
    pipeline,
    cache   : SemanticCache,
    top_k   : int = 5,
    verbose : bool = True,
) -> tuple[str, bool]:
    """
    Try the cache first; fall back to the full RAG pipeline on a miss.

    Returns
    -------
    (answer : str,  from_cache : bool)

    Example
    -------
    cache    = SemanticCache(threshold=0.92)
    pipeline = RAGPipeline().load()

    answer, from_cache = cached_rag_query("What is IFC net income?", pipeline, cache)
    print("(from cache)" if from_cache else "(freshly generated)")
    """
    if verbose:
        print(f"\n🔍 Query: '{query[:70]}...'")

    # Step 1: Try cache
    cached_answer, similarity = cache.lookup(query)
    if cached_answer is not None:
        return cached_answer, True  # from_cache = True

    # Step 2: Cache miss → run the full pipeline
    if verbose:
        print("  ⚙️  Generating answer via RAG pipeline...")

    t0     = time.perf_counter()
    chunks = pipeline.get_context(query, top_k=top_k)
    answer = pipeline.query(query, top_k=top_k)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  ⏱️  Generated in {elapsed:.2f}s")

    # Step 3: Store in cache for next time
    cache.store(query, answer, context_chunks=chunks)

    return answer, False  # from_cache = False
