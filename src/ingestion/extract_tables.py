"""
src/ingestion/extract_tables.py
────────────────────────────────
Practice 1 – Data Parsing Requirement 3
Extract ALL tables from the IFC PDF.

Strategy:
  1. pdfplumber  → finds table bounding boxes per page
  2. Table is saved as Markdown (human-readable) + JSON (machine-readable)
  3. Metadata: page number, table index on page, nearby heading text
"""

import json
from pathlib import Path

import pdfplumber
from tqdm import tqdm

from config.settings import PDF_PATH, DATA_TABLES


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list-of-lists) to a Markdown string."""
    if not table or not table[0]:
        return ""

    # Clean cells – replace None with ""
    rows = [[str(c).strip() if c else "" for c in row] for row in table]

    header = rows[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows[1:]:
        # Pad short rows
        while len(row) < len(header):
            row.append("")
        md += "| " + " | ".join(row) + " |\n"
    return md


def extract_tables_from_pdf(pdf_path: Path = PDF_PATH) -> list[dict]:
    """
    Returns a list of table dicts:
      - page_number   : int
      - table_index   : int  (0-based index on that page)
      - markdown      : str  (Markdown representation)
      - raw_data      : list (original 2-D list from pdfplumber)
      - rows          : int
      - cols          : int
      - source_file   : str
      - content_type  : "table"
    """
    all_tables = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting tables")):
            page_num = i + 1

            # Get text just above tables for context
            page_text = page.extract_text() or ""
            tables = page.extract_tables()

            for t_idx, table in enumerate(tables):
                if not table:
                    continue

                md = _table_to_markdown(table)
                if not md.strip():
                    continue

                table_dict = {
                    "page_number" : page_num,
                    "table_index" : t_idx,
                    "markdown"    : md,
                    "raw_data"    : table,
                    "rows"        : len(table),
                    "cols"        : len(table[0]) if table else 0,
                    "source_file" : pdf_path.name,
                    "content_type": "table",
                    # First 300 chars of page text as nearby-text hint
                    "nearby_text" : page_text[:300],
                }
                all_tables.append(table_dict)

    print(f"✅ Found {len(all_tables)} tables across the document")
    return all_tables


def save_extracted_tables(tables: list[dict], out_dir: Path = DATA_TABLES):
    """
    Save tables in two formats:
      1. tables.json       – full data with metadata
      2. tables_markdown/  – individual .md files (easy to inspect)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON (everything) ─────────────────────────────────────────────────
    json_file = out_dir / "tables.json"
    # Exclude raw_data from JSON to keep it smaller
    slim = [{k: v for k, v in t.items() if k != "raw_data"} for t in tables]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)

    # ── Markdown files (one per table) ────────────────────────────────────
    md_dir = out_dir / "tables_markdown"
    md_dir.mkdir(exist_ok=True)
    for t in tables:
        fname = f"page{t['page_number']:03d}_table{t['table_index']}.md"
        with open(md_dir / fname, "w", encoding="utf-8") as f:
            f.write(f"# Table – Page {t['page_number']}, Index {t['table_index']}\n\n")
            f.write(t["markdown"])

    print(f"💾 Saved tables → {json_file}")
    print(f"💾 Markdown files → {md_dir}")
    return json_file


if __name__ == "__main__":
    tables = extract_tables_from_pdf()
    save_extracted_tables(tables)
