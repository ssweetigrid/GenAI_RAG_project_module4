"""
scripts/run_phase5_multimodal.py
──────────────────────────────────
Phase 5: Multimodal RAG – Tables + Images

This script:
  1. Extracts tables from the PDF (pdfplumber)
  2. Extracts images + generates Gemini captions
  3. Combines text + table + image chunks
  4. Re-embeds everything and rebuilds the vector stores
  5. Tests queries that require table or image knowledge

Usage:
    python scripts/run_phase5_multimodal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.extract_tables       import extract_tables_from_pdf, save_extracted_tables
from src.ingestion.extract_images       import extract_images_from_pdf, save_extracted_images
from src.retrieval.multimodal_retriever import load_all_chunks_multimodal
from src.embeddings.embed               import embed_chunks, save_embeddings
from src.embeddings.faiss_store         import build_faiss_index
from src.embeddings.qdrant_store        import build_qdrant_collection, get_qdrant_client
from src.rag_pipeline                   import RAGPipeline

# Multimodal-specific queries
MULTIMODAL_QUERIES = [
    "What does the chart showing IFC's income trends depict?",
    "Show me the table of IFC's selected financial data for 2024 and 2023.",
    "What was IFC's net income according to the financial statements table?",
]


def main():
    print("=" * 60)
    print("  PHASE 5 – Multimodal RAG (Tables + Images)")
    print("=" * 60)

    # ── Step 1: Extract tables ────────────────────────────────────────────
    print("\n📊 Step 1: Extracting tables...")
    tables = extract_tables_from_pdf()
    save_extracted_tables(tables)

    # ── Step 2: Extract images + caption ─────────────────────────────────
    print("\n🖼️  Step 2: Extracting images and generating captions...")
    print("   (This calls Gemini for each image – may take a few minutes)")
    images = extract_images_from_pdf(caption=True)
    save_extracted_images(images)

    # ── Step 3: Load all chunks (text + tables + images) ──────────────────
    print("\n📦 Step 3: Loading all multimodal chunks...")
    all_chunks = load_all_chunks_multimodal()

    # ── Step 4: Embed everything ──────────────────────────────────────────
    print("\n🔢 Step 4: Embedding all chunks...")
    embedded = embed_chunks(all_chunks)
    save_embeddings(embedded)

    # ── Step 5: Rebuild vector stores ────────────────────────────────────
    print("\n🗃️  Step 5: Rebuilding vector stores with multimodal data...")
    build_faiss_index(embedded)

    try:
        client = get_qdrant_client()
        build_qdrant_collection(embedded, client)
    except Exception as e:
        print(f"⚠️  Qdrant: {e}")

    # ── Step 6: Test multimodal queries ──────────────────────────────────
    print("\n🧪 Step 6: Testing multimodal queries...")

    pipeline = RAGPipeline(retriever_mode="faiss").load(chunks=all_chunks)

    for query in MULTIMODAL_QUERIES:
        print(f"\n❓ {query}")

        # Show what type of content was retrieved
        chunks = pipeline.get_context(query, top_k=5)
        type_counts = {}
        for c in chunks:
            ct = c.get("content_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        print(f"   Retrieved: {type_counts}")

        # Generate answer
        print("💡 Answer:")
        for token in pipeline.query(query, stream=True):
            print(token, end="", flush=True)
        print()

    print("\n✅ Phase 5 complete!")


if __name__ == "__main__":
    main()
