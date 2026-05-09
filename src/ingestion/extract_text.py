"""
src/ingestion/extract_text.py
──────────────────────────────
Practice 1 – Data Parsing Requirement 1
Extract ALL textual content + metadata from the IFC PDF.

Libraries used:
  • pdfplumber  – text per page with bounding boxes
  • pypdf       – document-level metadata
"""

import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm

from config.settings import PDF_PATH, DATA_TEXT


def extract_text_from_pdf(pdf_path: Path = PDF_PATH) -> list[dict]:
    """
    Returns a list of page dicts, each containing:
      - page_number  : int
      - text         : str   (raw text)
      - char_count   : int
      - metadata     : dict  (author, title, creation date, etc.)
    """

    # ── Step 1: Pull document-level metadata from pypdf ──────────────────
    reader = PdfReader(str(pdf_path))
    doc_meta = {}
    if reader.metadata:
        doc_meta = {
            "title"   : reader.metadata.get("/Title", ""),
            "author"  : reader.metadata.get("/Author", ""),
            "subject" : reader.metadata.get("/Subject", ""),
            "creator" : reader.metadata.get("/Creator", ""),
            "producer": reader.metadata.get("/Producer", ""),
            "created" : str(reader.metadata.get("/CreationDate", "")),
        }
    total_pages = len(reader.pages)
    print(f"📄 PDF has {total_pages} pages")
    print(f"   Metadata: {doc_meta}")

    # ── Step 2: Extract text page-by-page using pdfplumber ───────────────
    pages_data = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting text")):
            text = page.extract_text() or ""

            page_dict = {
                "page_number": i + 1,           # 1-indexed
                "text"       : text.strip(),
                "char_count" : len(text),
                "doc_metadata": doc_meta,
                "source_file": pdf_path.name,
                "content_type": "text",
            }
            pages_data.append(page_dict)

    print(f"✅ Extracted text from {len(pages_data)} pages")
    return pages_data


def save_extracted_text(pages_data: list[dict], out_dir: Path = DATA_TEXT):
    """Save each page as a JSON file for later chunking."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all pages in one JSON file
    out_file = out_dir / "pages_text.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved text data → {out_file}")
    return out_file


if __name__ == "__main__":
    pages = extract_text_from_pdf()
    save_extracted_text(pages)
