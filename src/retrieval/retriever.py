"""
src/retrieval/retriever.py
───────────────────────────
Phase 1 + 3 – Unified Retrieval

Supports three retrieval modes:
  "faiss"  – dense vector search via FAISS  (Phase 1)
  "qdrant" – dense vector search via Qdrant (Phase 1)
  "hybrid" – BM25 sparse + FAISS dense, scores merged (Phase 3)

Cross-encoder re-ranking is available in all modes (Phase 3).
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import TOP_K
from config.gcp_auth import init_vertex
from src.embeddings.faiss_store import load_faiss_index, search_faiss
from src.embeddings.qdrant_store import get_qdrant_client, search_qdrant


# ── Module-level singletons (loaded once, reused) ─────────────────────────
_embedding_model = None
_cross_encoder   = None
_bm25            = None
_bm25_chunks     = None


def _get_embed_model():
    """Return the Vertex AI embedding model, loading it once."""
    global _embedding_model
    if _embedding_model is None:
        init_vertex()
        from vertexai.language_models import TextEmbeddingModel
        _embedding_model = TextEmbeddingModel.from_pretrained(
            __import__("config.settings", fromlist=["EMBEDDING_MODEL"]).EMBEDDING_MODEL
        )
    return _embedding_model


def _get_cross_encoder():
    """Load the cross-encoder re-ranking model (runs locally, CPU-only)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Cross-encoder loaded (ms-marco-MiniLM-L-6-v2)")
    return _cross_encoder


# ── Core embedding helper ─────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Embed a single query string using Vertex AI.
    Returns a plain Python list of 768 floats.
    """
    model   = _get_embed_model()
    results = model.get_embeddings([query])
    return list(results[0].values)   # .values → plain list, JSON-serialisable


# ── BM25 sparse retrieval (Phase 3) ──────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> None:
    """Build an in-memory BM25 index from text chunks."""
    global _bm25, _bm25_chunks
    tokenised  = [c["text"].lower().split() for c in chunks]
    _bm25      = BM25Okapi(tokenised)
    _bm25_chunks = chunks
    print(f"✅ BM25 index built  |  {len(chunks)} documents")


def search_bm25(query: str, top_k: int = TOP_K) -> list[dict]:
    """Keyword search using BM25. Returns top-k chunk dicts with a 'score' key."""
    if _bm25 is None:
        raise RuntimeError("Call build_bm25_index(chunks) before searching.")
    scores      = _bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        r = _bm25_chunks[idx].copy()
        r["score"] = float(scores[idx])
        results.append(r)
    return results


# ── Cross-encoder re-ranking (Phase 3) ───────────────────────────────────

def rerank(query: str, results: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Re-score retrieved chunks with a cross-encoder and return the top-k.

    The cross-encoder sees both the query and document text together,
    giving a much more accurate relevance score than cosine similarity alone.
    """
    ce     = _get_cross_encoder()
    pairs  = [(query, r["text"]) for r in results]
    scores = ce.predict(pairs)

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ── Unified Retriever class ───────────────────────────────────────────────

class Retriever:
    """
    Unified retriever.

    Usage
    -----
    r = Retriever(mode="faiss").load()
    chunks = r.retrieve("What was IFC net income?", top_k=5)
    """

    def __init__(self, mode: Literal["faiss", "qdrant", "hybrid"] = "faiss"):
        self.mode          = mode
        self._faiss_index  = None
        self._faiss_meta   = None
        self._qdrant       = None

    def load(self, chunks: list[dict] = None) -> "Retriever":
        """
        Load vector stores from disk.

        Parameters
        ----------
        chunks : required when mode="hybrid" (needed to build BM25 index)
        """
        if self.mode in ("faiss", "hybrid"):
            self._faiss_index, self._faiss_meta = load_faiss_index()

        if self.mode in ("qdrant", "hybrid"):
            try:
                self._qdrant = get_qdrant_client()
            except Exception as e:
                if self.mode == "qdrant":
                    raise
                print(f"⚠️  Qdrant unavailable, hybrid will use FAISS only: {e}")

        if self.mode == "hybrid" and chunks:
            build_bm25_index(chunks)

        print(f"✅ Retriever ready  |  mode={self.mode}")
        return self

    def retrieve(
        self,
        query      : str,
        top_k      : int  = TOP_K,
        use_rerank : bool = False,
        page_range : tuple = None,
        content_type: str = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query        : user question string
        top_k        : how many chunks to return
        use_rerank   : run cross-encoder re-ranking on the candidates
        page_range   : (min_page, max_page) filter – Qdrant only
        content_type : "text" | "table" | "image" filter – Qdrant only

        Returns
        -------
        list of chunk dicts, sorted by relevance descending
        """
        q_emb = embed_query(query)

        # Fetch more candidates when re-ranking so the reranker has room
        fetch_k = top_k * 3 if use_rerank else top_k * 2

        if self.mode == "faiss":
            results = search_faiss(q_emb, self._faiss_index,
                                   self._faiss_meta, fetch_k)

        elif self.mode == "qdrant":
            results = search_qdrant(
                q_emb, self._qdrant, fetch_k,
                page_range=page_range,
                content_type=content_type,
            )

        elif self.mode == "hybrid":
            dense  = search_faiss(q_emb, self._faiss_index,
                                  self._faiss_meta, top_k)
            sparse = search_bm25(query, top_k)

            # Merge by chunk_id, averaging scores
            merged: dict[str, dict] = {}
            for r in dense:
                merged[r["chunk_id"]] = {**r, "hybrid_score": r["score"]}
            for r in sparse:
                cid = r["chunk_id"]
                if cid in merged:
                    # Normalise BM25 scores to roughly the same range (0-1)
                    bm25_norm = r["score"] / (r["score"] + 1)
                    merged[cid]["hybrid_score"] = (
                        merged[cid]["hybrid_score"] + bm25_norm
                    ) / 2
                else:
                    bm25_norm = r["score"] / (r["score"] + 1)
                    merged[cid] = {**r, "hybrid_score": bm25_norm}

            results = list(merged.values())

            # Apply metadata filters in hybrid mode as well so Phase 3 can
            # experiment with the same controls as Qdrant retrieval.
            if page_range is not None:
                p_min, p_max = page_range
                results = [
                    r for r in results
                    if p_min <= int(r.get("page_number", -1)) <= p_max
                ]
            if content_type is not None:
                results = [
                    r for r in results
                    if r.get("content_type") == content_type
                ]

            results = sorted(
                results,
                key=lambda x: x["hybrid_score"],
                reverse=True,
            )
        else:
            raise ValueError(f"Unknown retriever mode: {self.mode}")

        if use_rerank:
            results = rerank(query, list(results), top_k)
        else:
            results = list(results)[:top_k]

        # Final safety gate: always enforce metadata filters on returned items.
        if page_range is not None:
            p_min, p_max = page_range
            results = [
                r for r in results
                if p_min <= int(r.get("page_number", -1)) <= p_max
            ]
        if content_type is not None:
            results = [r for r in results if r.get("content_type") == content_type]

        return results
