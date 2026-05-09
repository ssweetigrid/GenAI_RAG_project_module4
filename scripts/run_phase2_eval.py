"""
scripts/run_phase2_eval.py
───────────────────────────
Phase 2: RAG Evaluation

Evaluates the pipeline using:
  • RAGAS  (faithfulness, answer_relevancy, context_recall, context_precision)
  • LLM-as-a-Judge  (Gemini scoring 1–5 per question)

Then prints a comparison table of ALL saved evaluation runs so you can
see how each phase changed the scores.

Usage:
    python scripts/run_phase2_eval.py --mode faiss
    python scripts/run_phase2_eval.py --mode qdrant
    python scripts/run_phase2_eval.py --mode hybrid --rerank
    python scripts/run_phase2_eval.py --mode faiss --top_k 3
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ragas_eval import (
    load_eval_dataset,
    run_pipeline_on_eval_set,
    run_ragas_evaluation,
    run_llm_judge_evaluation,
    save_evaluation_results,
)
from src.evaluation.compare_phases import compare_all_phases
from src.rag_pipeline              import RAGPipeline
from src.ingestion.chunk_text      import load_chunks


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation – Phase 2")
    parser.add_argument("--mode",   default="faiss",
                        choices=["faiss", "qdrant", "hybrid"])
    parser.add_argument("--rerank", action="store_true",
                        help="Enable cross-encoder re-ranking")
    parser.add_argument("--top_k",  type=int, default=5,
                        help="Number of chunks to retrieve per question")
    args = parser.parse_args()

    label = f"{args.mode}" + ("_rerank" if args.rerank else "") + f"_k{args.top_k}"

    print("=" * 60)
    print(f"  PHASE 2 – RAG Evaluation  |  {label}")
    print("=" * 60)

    # ── Load pipeline ─────────────────────────────────────────────────────
    print("\n⚙️  Loading pipeline …")
    chunks   = load_chunks()
    pipeline = RAGPipeline(
        retriever_mode=args.mode,
        use_rerank=args.rerank,
    ).load(chunks=chunks)

    # ── Load eval dataset ─────────────────────────────────────────────────
    eval_df = load_eval_dataset()

    # ── Run pipeline on all eval questions ────────────────────────────────
    print(f"\n🔍 Running pipeline on {len(eval_df)} questions …")
    results = run_pipeline_on_eval_set(pipeline, eval_df, top_k=args.top_k)

    # ── RAGAS evaluation ──────────────────────────────────────────────────
    try:
        ragas_scores = run_ragas_evaluation(results)
    except Exception as e:
        print(f"\n⚠️  RAGAS evaluation skipped: {e}")
        ragas_scores = {}

    # ── LLM-as-a-Judge ────────────────────────────────────────────────────
    judge_df = run_llm_judge_evaluation(results)

    # ── Save results ──────────────────────────────────────────────────────
    save_evaluation_results(ragas_scores, judge_df, pipeline_name=label)

    # ── Cross-phase comparison ────────────────────────────────────────────
    print("\n📊 Cross-phase comparison:")
    compare_all_phases()

    print(f"\n✅ Evaluation complete!  results saved as eval_ragas_{label}.json")


if __name__ == "__main__":
    main()
