"""
src/ingestion/chunk_text.py
────────────────────────────
Phase 1, Task 1b – Text Cleaning & Chunking
Reads the extracted pages JSON and splits them into overlapping chunks
that are the right size for embedding.

Strategy:
  • RecursiveCharacterTextSplitter from LangChain
  • Each chunk carries metadata: page_number, source_file, content_type
"""

import json
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import DATA_TEXT, CHUNK_SIZE, CHUNK_OVERLAP


def load_pages(pages_json: Path = None) -> list[dict]:
    """Load the pages_text.json produced by extract_text.py"""
    if pages_json is None:
        pages_json = DATA_TEXT / "pages_text.json"

    with open(pages_json, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Remove excessive whitespace
      - Remove hyphenation at line breaks
      - Keep the text readable
    """
    # Join lines that were broken mid-word (PDF hyphenation)
    text = text.replace("-\n", "")
    # Collapse multiple spaces/newlines
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split pages into overlapping chunks.

    Returns a list of chunk dicts:
      - chunk_id     : str  (unique id)
      - text         : str  (chunk content)
      - page_number  : int  (source page)
      - source_file  : str
      - content_type : "text"
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators try to split at paragraph > sentence > word level
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_id = 0

    for page in pages:
        raw_text = page.get("text", "")
        if not raw_text.strip():
            continue  # Skip blank pages

        clean = clean_text(raw_text)
        # Split this page's text into sub-chunks
        sub_chunks = splitter.split_text(clean)

        for sub in sub_chunks:
            if len(sub.strip()) < 50:
                continue  # Skip very short chunks

            chunks.append({
                "chunk_id"    : f"chunk_{chunk_id:05d}",
                "text"        : sub.strip(),
                "page_number" : page["page_number"],
                "source_file" : page["source_file"],
                "content_type": "text",
                "char_count"  : len(sub),
            })
            chunk_id += 1

    print(f"✅ Created {len(chunks)} text chunks  (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def save_chunks(chunks: list[dict], out_dir: Path = DATA_TEXT):
    """Save chunks to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved chunks → {out_file}")
    return out_file


def load_chunks(chunks_json: Path = None) -> list[dict]:
    """Load chunks.json"""
    if chunks_json is None:
        chunks_json = DATA_TEXT / "chunks.json"
    with open(chunks_json, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    pages = load_pages()
    chunks = chunk_pages(pages)
    save_chunks(chunks)
