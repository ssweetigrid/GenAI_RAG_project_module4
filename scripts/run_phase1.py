"""
scripts/run_phase1.py
──────────────────────
Phase 1: Text-Based RAG System

Runs the complete Phase 1 pipeline:
  1. Extract text from PDF           (pdfplumber + pypdf)
  2. Clean and chunk text            (LangChain splitter)
  3. Generate dense embeddings       (Vertex AI text-embedding-004)
  4. Build FAISS in-memory index
  5. Build Qdrant persistent index   (requires Docker)
  6. Test a sample query end-to-end

Usage:
    python scripts/run_phase1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.extract_text  import extract_text_from_pdf, save_extracted_text
from src.ingestion.chunk_text    import chunk_pages, save_chunks, load_pages
from src.embeddings.embed        import embed_chunks, save_embeddings
from src.embeddings.faiss_store  import build_faiss_index
from src.embeddings.qdrant_store import build_qdrant_collection, get_qdrant_client
from src.rag_pipeline            import RAGPipeline


def main():
    print("=" * 60)
    print("  PHASE 1 – Text-Based RAG System")
    print("=" * 60)

    # Step 1 – Extract text
    print("\n📖 Step 1: Extracting text from PDF …")
    pages = extract_text_from_pdf()
    save_extracted_text(pages)

    # Step 2 – Chunk
    print("\n✂️  Step 2: Chunking text …")
    pages_loaded = load_pages()
    chunks       = chunk_pages(pages_loaded)
    save_chunks(chunks)

    # Step 3 – Embed
    print("\n🔢 Step 3: Generating embeddings (calls Vertex AI) …")
    embedded = embed_chunks(chunks)
    save_embeddings(embedded)

    # Step 4 – FAISS
    print("\n🗃️  Step 4: Building FAISS index …")
    build_faiss_index(embedded)

    # Step 5 – Qdrant (optional – needs Docker)
    print("\n🗃️  Step 5: Building Qdrant collection …")
    try:
        client = get_qdrant_client()
        build_qdrant_collection(embedded, client)
    except Exception as e:
        print(f"⚠️  Qdrant skipped: {e}")
        print("   Start Docker first: cd docker && docker-compose up -d")

    # Step 6 – Test
    print("\n🧪 Step 6: Testing the pipeline …")
    pipeline = RAGPipeline(retriever_mode="faiss").load(chunks=chunks)

    test_q = "What was IFC's net income for fiscal year 2024?"
    print(f"\n❓ {test_q}")
    print("💡 Answer:")
    for token in pipeline.query(test_q, stream=True):
        print(token, end="", flush=True)
    print("\n")

    print("=" * 60)
    print("  Phase 1 Complete! ✅")
    print("=" * 60)
    print("\nNext steps:")
    print("  Evaluate:   python scripts/run_phase2_eval.py --mode faiss")
    print("  UI:         streamlit run src/ui/streamlit_app.py")
    print("  Compare:    python scripts/compare_vector_stores.py")


if __name__ == "__main__":
    main()
