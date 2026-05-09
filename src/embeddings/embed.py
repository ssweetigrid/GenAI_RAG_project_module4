"""
src/embeddings/embed.py
────────────────────────
Phase 1, Task 2 – Embedding Generation

Generate 768-dim dense vectors for every text chunk using
the Vertex AI text-embedding-004 model.

Output saved to:  data/processed/text/embeddings.json
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from config.settings import EMBEDDING_MODEL, DATA_TEXT, ROOT_DIR
from config.gcp_auth import init_vertex

# Vertex AI text-embedding-004 accepts up to 5 texts per call
BATCH_SIZE = 5


def get_embedding_model():
    """Initialise Vertex AI and return the embedding model instance."""
    init_vertex()
    # Use the stable vertexai.language_models path
    from vertexai.language_models import TextEmbeddingModel
    return TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)


def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    """
    Embed a list of strings.  Returns a list of float lists.
    Each float list has 768 dimensions.
    """
    if model is None:
        model = get_embedding_model()
    results = model.get_embeddings(texts)
    # .values returns a sequence – convert to plain Python list for JSON
    return [list(r.values) for r in results]


def embed_chunks(
    chunks: list[dict],
    model=None,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Generate embeddings for all chunks.

    Each returned dict has the same fields as the input chunk plus:
      - embedding : list[float]  (768 dimensions, plain Python list)
    """
    if model is None:
        model = get_embedding_model()

    embedded = []

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            vectors = embed_texts(texts, model)
            for chunk, vec in zip(batch, vectors):
                item = {
                    "chunk_id"    : chunk["chunk_id"],
                    "embedding"   : vec,          # plain list[float] – JSON-safe
                    "text"        : chunk["text"],
                    "page_number" : chunk["page_number"],
                    "source_file" : chunk["source_file"],
                    "content_type": chunk["content_type"],
                }
                # Preserve optional metadata used by multimodal/visual phases.
                for k in ("image_path", "file_path", "description", "table_index"):
                    if k in chunk:
                        item[k] = chunk[k]
                embedded.append(item)
        except Exception as e:
            print(f"⚠️  Batch {i // batch_size} failed: {e}")
            time.sleep(2)

        # Polite pause to avoid rate-limit errors
        time.sleep(0.2)

    print(f"✅ Embedded {len(embedded)} chunks  (dim=768)")
    return embedded


def save_embeddings(embedded: list[dict], out_dir: Path = DATA_TEXT) -> Path:
    """Save embeddings list to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "embeddings.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(embedded, f)
    print(f"💾 Saved embeddings → {out_file}")
    return out_file


def load_embeddings(emb_json: Path = None) -> list[dict]:
    """Load embeddings from JSON."""
    if emb_json is None:
        emb_json = DATA_TEXT / "embeddings.json"
    if not Path(emb_json).exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_json}\n"
            "Run `python scripts/run_phase1.py` first."
        )
    with open(emb_json, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.ingestion.chunk_text import load_chunks
    chunks   = load_chunks()
    embedded = embed_chunks(chunks)
    save_embeddings(embedded)
