"""
src/embeddings/qdrant_store.py
────────────────────────────────
Phase 1, Task 2 – Qdrant Vector Store
Build a Qdrant collection from embeddings and provide search.

Qdrant is a production-grade vector database with rich filtering.
It runs locally via Docker (see docker/ folder).
"""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition,
    Range, MatchValue,
)
from tqdm import tqdm

from config.settings import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
from src.embeddings.embed import load_embeddings


def get_qdrant_client() -> QdrantClient:
    """Connect to the locally-running Qdrant instance."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"✅ Connected to Qdrant  |  {QDRANT_HOST}:{QDRANT_PORT}")
    return client


def build_qdrant_collection(
    embedded: list[dict],
    client: QdrantClient = None,
    collection_name: str = QDRANT_COLLECTION,
):
    """
    Create (or recreate) a Qdrant collection and upload all embeddings.

    Each point has:
      - vector   : list[float]  (768-dim)
      - payload  : dict         (chunk_id, text, page_number, content_type)
    """
    if client is None:
        client = get_qdrant_client()

    dim = len(embedded[0]["embedding"])

    # Delete old collection if it exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"🗑️  Deleted old collection '{collection_name}'")

    # Create fresh collection with cosine similarity
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"✅ Created collection '{collection_name}'  dim={dim}")

    # Upload points in batches
    BATCH = 100
    for i in tqdm(range(0, len(embedded), BATCH), desc="Uploading to Qdrant"):
        batch = embedded[i : i + BATCH]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=e["embedding"],
                payload={
                    "chunk_id"    : e["chunk_id"],
                    "text"        : e["text"],
                    "page_number" : e["page_number"],
                    "source_file" : e["source_file"],
                    "content_type": e["content_type"],
                },
            )
            for e in batch
        ]
        client.upsert(collection_name=collection_name, points=points)

    count = client.count(collection_name).count
    print(f"✅ Qdrant collection ready  |  {count} points")
    return client


def search_qdrant(
    query_embedding: list[float],
    client: QdrantClient,
    top_k: int = 5,
    collection_name: str = QDRANT_COLLECTION,
    page_range: tuple = None,         # e.g. (10, 20) to restrict pages
    content_type: str = None,         # e.g. "text" or "table"
) -> list[dict]:
    """
    Search Qdrant for the top-k nearest chunks.

    Supports metadata filtering:
      - page_range   : only search within certain pages
      - content_type : "text", "table", or "image"
    """
    # Build optional filter
    must_conditions = []
    if page_range:
        must_conditions.append(
            FieldCondition(
                key="page_number",
                range=Range(gte=page_range[0], lte=page_range[1]),
            )
        )
    if content_type:
        must_conditions.append(
            FieldCondition(
                key="content_type",
                match=MatchValue(value=content_type),
            )
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    # qdrant-client API differs by version. Prefer query_points first.
    try:
        query_resp = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = getattr(query_resp, "points", query_resp)
    except AttributeError:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

    results = []
    for hit in hits:
        r = hit.payload.copy()
        r["score"] = hit.score
        results.append(r)

    return results


if __name__ == "__main__":
    embedded = load_embeddings()
    client = get_qdrant_client()
    build_qdrant_collection(embedded, client)
