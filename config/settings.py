"""
config/settings.py
──────────────────
Central place to load all environment variables.
Every other module imports from here – never import dotenv elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Google Cloud ──────────────────────────────────────────────────────────
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "keys/sweeti.json")
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_LOCATION    = os.getenv("GCP_LOCATION", "us-central1")

# ── Models ────────────────────────────────────────────────────────────────
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# ── Qdrant ────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ifc_rag")

# ── Langfuse ──────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── RAG tuning ───────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K         = int(os.getenv("TOP_K", 5))

# ── Paths (always relative to project root) ───────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_TEXT      = ROOT_DIR / "data" / "processed" / "text"
DATA_TABLES    = ROOT_DIR / "data" / "processed" / "tables"
DATA_IMAGES    = ROOT_DIR / "data" / "processed" / "images"
FAISS_INDEX_PATH = str(ROOT_DIR / "data" / "processed" / "faiss_index")
PDF_PATH       = DATA_RAW / "ifc-annual-report-2024-financials.pdf"
EVAL_XLSX_PATH = DATA_RAW / "RAG_evaluation_dataset.xlsx"
