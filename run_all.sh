#!/usr/bin/env bash
# ============================================================
#  run_all.sh  –  Full pipeline Phase 1 → 2 → 3 → 4
#
#  Phases 5 and 6 are commented out (slow / optional).
#  Uncomment them to run the full multimodal pipeline.
# ============================================================
set -euo pipefail     # stop on any error; treat unset vars as errors

SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         IFC RAG PROJECT – Full Pipeline Run         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────
echo "🔍 Pre-flight checks …"

if [ ! -f "keys/sweeti.json" ]; then
  echo ""
  echo "❌  ERROR: keys/sweeti.json not found!"
  echo "   Copy your GCP service-account key:"
  echo "     cp /path/to/sweeti.json keys/sweeti.json"
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "⚠️   .env not found – copying from .env.example"
  cp .env.example .env
  echo ""
  echo "   ➡️  Open .env and set GCP_PROJECT_ID=your-actual-project-id"
  echo "   Then re-run this script."
  exit 1
fi

# Quick check that GCP_PROJECT_ID was actually set
if grep -q "your-gcp-project-id" .env; then
  echo ""
  echo "❌  ERROR: GCP_PROJECT_ID is still the placeholder value."
  echo "   Edit .env and set your real project ID."
  exit 1
fi

echo "✅ keys/sweeti.json found"
echo "✅ .env configured"
echo ""

# Resolve Docker Compose command for both legacy/new CLIs.
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
else
  echo "❌  ERROR: Docker is not installed or not on PATH."
  echo "   Install Docker Desktop and retry."
  exit 1
fi

# ── Docker services ───────────────────────────────────────────────────────
echo "🐳 Starting Docker services (Qdrant + Langfuse) …"
cd docker && $COMPOSE_CMD up -d
cd ..
echo "   Waiting 5s for services to initialise …"
sleep 5
echo ""

# ── Phase 1 ───────────────────────────────────────────────────────────────
echo "$SEP"
echo "  PHASE 1 – Text RAG"
echo "$SEP"
python scripts/run_phase1.py
echo ""

# ── Phase 2 (baseline) ────────────────────────────────────────────────────
echo "$SEP"
echo "  PHASE 2 – Evaluation (FAISS baseline)"
echo "$SEP"
python scripts/run_phase2_eval.py --mode faiss --top_k 5
echo ""

# ── Phase 3 ───────────────────────────────────────────────────────────────
echo "$SEP"
echo "  PHASE 3 – Hybrid Search & Re-ranking"
echo "$SEP"
python scripts/run_phase3_hybrid.py
echo ""

# ── Phase 2 re-evaluation after Phase 3 ──────────────────────────────────
echo "  Re-evaluating with Hybrid + Re-rank …"
python scripts/run_phase2_eval.py --mode hybrid --rerank --top_k 5
echo ""

# ── Phase 4 ───────────────────────────────────────────────────────────────
echo "$SEP"
echo "  PHASE 4 – Semantic Cache + Multi-hop"
echo "$SEP"
python scripts/run_phase4_advanced.py --task both
echo ""

# ── Phase 5 (optional – uncomment to run) ─────────────────────────────────
# echo "$SEP"
# echo "  PHASE 5 – Multimodal RAG (Tables + Images)"
# echo "$SEP"
# python scripts/run_phase5_multimodal.py
# python scripts/run_phase2_eval.py --mode faiss --top_k 5

# ── Phase 6 (optional – very slow, one Gemini call per page) ──────────────
# echo "$SEP"
# echo "  PHASE 6 – ColPali Visual RAG"
# echo "$SEP"
# python scripts/run_phase6_colpali.py

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ All phases complete!                             ║"
echo "║                                                      ║"
echo "║  Streamlit UI:                                      ║"
echo "║    streamlit run src/ui/streamlit_app.py            ║"
echo "║    → http://localhost:8501                          ║"
echo "║                                                      ║"
echo "║  Qdrant dashboard  → http://localhost:6333/dashboard║"
echo "║  Langfuse tracing  → http://localhost:3000          ║"
echo "╚══════════════════════════════════════════════════════╝"
