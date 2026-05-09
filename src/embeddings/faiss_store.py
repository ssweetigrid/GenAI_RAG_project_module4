"""
src/embeddings/faiss_store.py
──────────────────────────────
Phase 1, Task 2 – FAISS Vector Store

Builds an in-memory FAISS index from embeddings and provides
fast cosine-similarity search.

Files saved to disk:
  data/processed/faiss_index.faiss   – binary FAISS index
  data/processed/faiss_index.pkl     – chunk metadata list
"""

import pickle
from pathlib import Path

import numpy as np
import faiss

from config.settings import ROOT_DIR

# Always use an absolute path so scripts work from any directory
_DEFAULT_INDEX_PATH = str(ROOT_DIR / "data" / "processed" / "faiss_index")


def build_faiss_index(
    embedded: list[dict],
    index_path: str = _DEFAULT_INDEX_PATH,
) -> tuple:
    """
    Build a FAISS flat inner-product index from embedded chunks.

    Parameters
    ----------
    embedded   : list of dicts with "embedding", "chunk_id", "text",
                 "page_number", "source_file", "content_type" fields
    index_path : base path (no extension); .faiss and .pkl are appended

    Returns
    -------
    (faiss.Index, list[dict])  – the index and the metadata list
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # Build (N, dim) float32 matrix
    vectors = np.array([e["embedding"] for e in embedded], dtype="float32")
    dim     = vectors.shape[1]

    # L2-normalise so inner-product == cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save index binary
    faiss.write_index(index, index_path + ".faiss")

    # Save metadata (everything except the raw embedding vector)
    meta = []
    for e in embedded:
        row = {
            "chunk_id"    : e["chunk_id"],
            "text"        : e["text"],
            "page_number" : e["page_number"],
            "source_file" : e["source_file"],
            "content_type": e["content_type"],
        }
        # Keep optional metadata for multimodal/visual retrieval.
        for k in ("image_path", "file_path", "description", "table_index"):
            if k in e:
                row[k] = e[k]
        meta.append(row)
    with open(index_path + ".pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ FAISS index built  |  {index.ntotal} vectors  dim={dim}")
    print(f"💾 Saved → {index_path}.faiss  &  {index_path}.pkl")
    return index, meta


def load_faiss_index(index_path: str = _DEFAULT_INDEX_PATH) -> tuple:
    """
    Load a previously saved FAISS index and its metadata.

    Returns
    -------
    (faiss.Index, list[dict])
    """
    faiss_file = index_path + ".faiss"
    meta_file  = index_path + ".pkl"

    if not Path(faiss_file).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_file}\n"
            "Run `python scripts/run_phase1.py` first to build the index."
        )

    index = faiss.read_index(faiss_file)
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)

    print(f"✅ FAISS index loaded  |  {index.ntotal} vectors")
    return index, meta


def search_faiss(
    query_embedding: list[float],
    index,
    meta: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Search the FAISS index for the top-k closest chunks.

    Returns a list of metadata dicts, each with an added 'score' key
    (cosine similarity, 0.0 – 1.0, higher = more relevant).
    """
    q = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(q)

    scores, indices = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # fewer results than top_k
        result = meta[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


if __name__ == "__main__":
    from src.embeddings.embed import load_embeddings
    embedded = load_embeddings()
    build_faiss_index(embedded)
