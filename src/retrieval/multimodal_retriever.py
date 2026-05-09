"""
src/retrieval/multimodal_retriever.py
──────────────────────────────────────
Phase 5 – Multimodal RAG
Extends retrieval to include tables and image captions.

Strategy:
  • Text chunks     → already in FAISS/Qdrant from Phase 1
  • Table chunks    → convert each table's markdown to a chunk + embed it
  • Image chunks    → convert each image caption to a chunk + embed it
  • All go into the SAME vector store (unified search space)
  • content_type field is used to filter if needed
"""

import json
from pathlib import Path

from config.settings import DATA_TABLES, DATA_IMAGES, DATA_TEXT


def load_table_chunks() -> list[dict]:
    """
    Load extracted tables and convert each to a chunk-like dict
    so they can be embedded just like text chunks.
    """
    tables_json = DATA_TABLES / "tables.json"
    if not tables_json.exists():
        print("⚠️  No tables.json found. Run extract_tables.py first.")
        return []

    with open(tables_json, "r", encoding="utf-8") as f:
        tables = json.load(f)

    chunks = []
    for i, t in enumerate(tables):
        # The text for embedding = Markdown table + nearby context
        text = f"Table from page {t['page_number']}:\n{t['markdown']}"
        if t.get("nearby_text"):
            text = t["nearby_text"][:200] + "\n\n" + text

        chunks.append({
            "chunk_id"    : f"table_{i:04d}",
            "text"        : text,
            "page_number" : t["page_number"],
            "source_file" : t["source_file"],
            "content_type": "table",
            "char_count"  : len(text),
        })

    print(f"✅ Loaded {len(chunks)} table chunks")
    return chunks


def load_image_chunks() -> list[dict]:
    """
    Load image captions and convert each to a chunk-like dict.
    """
    images_json = DATA_IMAGES / "images_metadata.json"
    if not images_json.exists():
        print("⚠️  No images_metadata.json found. Run extract_images.py first.")
        return []

    with open(images_json, "r", encoding="utf-8") as f:
        images = json.load(f)

    chunks = []
    for i, img in enumerate(images):
        cap = img.get("caption", "")
        if not cap or cap.startswith("[Caption error"):
            continue  # Skip images without valid captions

        text = f"Image/chart from page {img['page_number']}:\n{cap}"

        chunks.append({
            "chunk_id"    : f"image_{i:04d}",
            "text"        : text,
            "page_number" : img["page_number"],
            "source_file" : img["source_file"],
            "content_type": "image",
            "file_path"   : img.get("file_path", ""),
            "char_count"  : len(text),
        })

    print(f"✅ Loaded {len(chunks)} image chunks")
    return chunks


def load_all_chunks_multimodal() -> list[dict]:
    """
    Load text + table + image chunks together.
    This is the full multimodal corpus for Phase 5.
    """
    # Load text chunks
    text_json = DATA_TEXT / "chunks.json"
    if text_json.exists():
        with open(text_json, "r", encoding="utf-8") as f:
            text_chunks = json.load(f)
    else:
        text_chunks = []
        print("⚠️  No text chunks found. Run chunk_text.py first.")

    table_chunks = load_table_chunks()
    image_chunks = load_image_chunks()

    all_chunks = text_chunks + table_chunks + image_chunks
    print(f"\n📦 Total multimodal chunks: {len(all_chunks)}")
    print(f"   Text:   {len(text_chunks)}")
    print(f"   Tables: {len(table_chunks)}")
    print(f"   Images: {len(image_chunks)}")

    return all_chunks
