# 📄 IFC Annual Report 2024 – RAG System

A complete **Retrieval-Augmented Generation (RAG)** pipeline built on top of the IFC 2024 Annual Report PDF, powered by **Gemini 2.0 Flash** via **Vertex AI**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- GCP project with Vertex AI enabled
- Your `sweeti.json` service-account key

### 1. Setup
```bash
# Clone / extract the project
cd rag_project

# Put your GCP key in the keys folder
cp /path/to/sweeti.json keys/sweeti.json

# Create environment file
cp .env.example .env
# → Open .env and set GCP_PROJECT_ID=your-project-id

# Install Python dependencies
pip install -r requirements.txt

# Start Docker services (Qdrant + Langfuse)
cd docker && docker-compose up -d && cd ..
```

### 2. Run All Phases
```bash
# One-shot runner (Phase 1 → 3 + evaluations)
bash run_all.sh

# OR run phases individually:
python scripts/run_phase1.py
python scripts/run_phase2_eval.py --mode faiss
python scripts/run_phase3_hybrid.py
python scripts/run_phase5_multimodal.py   # Tables + Images
python scripts/run_phase6_colpali.py      # Visual RAG
```

### 3. Launch the UI
```bash
# Streamlit (recommended)
streamlit run src/ui/streamlit_app.py
# → http://localhost:8501

# OR Gradio
python src/ui/gradio_app.py
# → http://localhost:7860
```

### 4. Compare Vector Stores
```bash
python scripts/compare_vector_stores.py
```

---

## 📁 Project Structure

```
rag_project/
├── keys/sweeti.json          ← YOUR GCP KEY (never commit this!)
├── .env                      ← Environment variables
├── requirements.txt
├── Dockerfile
├── run_all.sh                ← One-shot runner
│
├── config/
│   ├── settings.py           ← All configuration in one place
│   └── gcp_auth.py           ← Vertex AI authentication
│
├── src/
│   ├── ingestion/            ← PDF → text, tables, images
│   ├── embeddings/           ← Vertex AI embeddings, FAISS, Qdrant
│   ├── retrieval/            ← Retriever, BM25, reranker, cache
│   ├── generation/           ← Gemini answer generation (streaming)
│   ├── evaluation/           ← RAGAS + LLM-as-a-Judge
│   ├── ui/                   ← Streamlit + Gradio apps
│   └── rag_pipeline.py       ← Main pipeline orchestrator
│
├── scripts/                  ← One script per phase
├── notebooks/                ← Jupyter notebooks for exploration
├── docker/                   ← Docker Compose files
└── data/
    ├── raw/                  ← PDF + evaluation XLSX
    └── processed/            ← Embeddings, indexes, captions
```

---

## 🗺️ Phases Overview

| Phase | What it builds | Script |
|-------|---------------|--------|
| Data Parsing | Text, table, image extraction | (part of Phase 1 & 5) |
| Phase 1 | Text RAG + FAISS + Qdrant + Streamlit UI | `run_phase1.py` |
| Phase 2 | RAGAS eval + LLM-as-a-Judge | `run_phase2_eval.py` |
| Phase 3 | BM25 + Hybrid + Cross-encoder reranking | `run_phase3_hybrid.py` |
| Phase 4 | Semantic caching | (integrated into pipeline) |
| Phase 5 | Multimodal: tables + image captions | `run_phase5_multimodal.py` |
| Phase 6 | ColPali-style visual page retrieval | `run_phase6_colpali.py` |

---

## 🛠️ Services

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit UI | http://localhost:8501 | Main query interface |
| Gradio UI | http://localhost:7860 | Alternative interface |
| Qdrant | http://localhost:6333/dashboard | Vector DB dashboard |
| Langfuse | http://localhost:3000 | Observability & tracing |

---

## 🔑 Key Concepts

- **RAG** = Find relevant text chunks → give them to Gemini → generate answer
- **FAISS** = Fast in-memory vector search (great for development)
- **Qdrant** = Production vector DB with metadata filtering
- **BM25** = Keyword-based search (complements embedding search)
- **Re-ranking** = Cross-encoder model re-scores initial results for better precision
- **RAGAS** = Framework to automatically evaluate RAG quality
- **ColPali** = Treat whole PDF pages as images → retrieve by visual content

---

## ⚙️ Configuration

Edit `.env` to change:
```bash
GCP_PROJECT_ID=your-project-id     # Required!
GEMINI_MODEL=gemini-2.0-flash-001
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
```
