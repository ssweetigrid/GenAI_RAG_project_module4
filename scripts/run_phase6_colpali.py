"""
scripts/run_phase6_colpali.py
───────────────────────────────
Phase 6: ColPali-like Visual RAG

Steps:
  1. Rasterise every PDF page to PNG (150 DPI)
  2. Gemini Vision describes each page in detail
  3. Embed the descriptions → build a separate FAISS index
  4. At query time: retrieve top-K page images → pass to Gemini Vision
  5. Compare results vs Phase 1-5 text/table/caption approach

WARNING: Step 2 makes one Gemini API call per page.
         For a 100-page document this takes ~5-10 minutes and costs API quota.

Usage:
    python scripts/run_phase6_colpali.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings               import ROOT_DIR
from src.retrieval.colpali_retriever import (
    pdf_pages_to_images,
    generate_page_descriptions,
    answer_with_visual_context,
    load_page_descriptions,
)
from src.embeddings.embed        import embed_chunks, get_embedding_model
from src.embeddings.faiss_store  import build_faiss_index, load_faiss_index, search_faiss
from src.retrieval.retriever     import embed_query

# Use an absolute path so this works from any directory
COLPALI_INDEX_PATH = str(ROOT_DIR / "data" / "processed" / "faiss_colpali")

TEST_QUERIES = [
    "What does the financial highlights chart show about IFC's income?",
    "What was IFC's net income according to the consolidated balance sheet?",
    "Describe the portfolio composition chart in the report.",
]


def build_colpali_index(descriptions: list[dict]) -> tuple:
    """
    Embed page descriptions and build a dedicated FAISS index.
    Stored separately so it does not overwrite the Phase 1 text index.
    """
    print("\n🔢 Embedding page descriptions …")
    model = get_embedding_model()

    # Reuse the embed_chunks helper – descriptions already look like chunks
    embedded = embed_chunks(descriptions, model=model)

    index, meta = build_faiss_index(embedded, index_path=COLPALI_INDEX_PATH)
    return index, meta


def retrieve_pages(query: str, index, meta, top_k: int = 3) -> list[dict]:
    """Embed query and find the top-k most relevant page dicts."""
    q_emb = embed_query(query)
    return search_faiss(q_emb, index, meta, top_k=top_k)


def main():
    print("=" * 60)
    print("  PHASE 6 – ColPali-like Visual RAG")
    print("=" * 60)

    # ── Step 1: Rasterise pages ───────────────────────────────────────────
    print("\n🖼️  Step 1: Rasterising PDF pages …")
    pages = pdf_pages_to_images(dpi=150)

    # ── Step 2: Generate Gemini page descriptions ─────────────────────────
    print("\n🤖 Step 2: Generating Gemini descriptions (slow – ~1 call/page) …")
    descriptions = generate_page_descriptions(pages)

    # ── Step 3: Embed and index ───────────────────────────────────────────
    print("\n🗃️  Step 3: Building ColPali FAISS index …")
    index, meta = build_colpali_index(descriptions)

    # ── Step 4: Test visual queries ───────────────────────────────────────
    print("\n🧪 Step 4: Testing visual queries …")

    for query in TEST_QUERIES:
        print(f"\n{'─'*60}")
        print(f"❓ {query}")

        top_pages = retrieve_pages(query, index, meta, top_k=3)
        print(f"   Top pages: {[p['page_number'] for p in top_pages]}")

        print("💡 Answer (from page images via Gemini Vision):")
        answer = answer_with_visual_context(query, top_pages)
        print(answer)

    # ── Step 5: Comparison notes ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase Comparison")
    print("=" * 60)
    print("""
  Phase 1–5  Text / table / caption embeddings
             → fast, cheap, good for text-heavy questions
             → chart answers rely on Gemini-generated captions (indirect)

  Phase 6    Full page images sent directly to Gemini Vision
             → slower & more expensive (image tokens)
             → better for chart/graph/layout questions
             → direct visual grounding, no information lost in captioning
    """)

    print("✅ Phase 6 complete!")


if __name__ == "__main__":
    main()
