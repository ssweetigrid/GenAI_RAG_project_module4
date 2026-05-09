"""
src/evaluation/ragas_eval.py
─────────────────────────────
Phase 2 – RAG Evaluation using RAGAS framework

Metrics evaluated:
  • faithfulness        – is the answer grounded in the context?
  • answer_relevancy    – is the answer relevant to the question?
  • context_recall      – did retrieval find the right chunks?
  • context_precision   – are retrieved chunks actually useful?

Also includes an LLM-as-a-Judge scorer for nuanced evaluation.
"""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from google import genai
from google.genai import types

from config.settings import (
    EVAL_XLSX_PATH, DATA_TEXT,
    GEMINI_MODEL, EMBEDDING_MODEL, GCP_PROJECT_ID, GCP_LOCATION,
)
from config.gcp_auth import init_vertex
from src.generation.generator import generate_answer


# ── Load evaluation dataset ───────────────────────────────────────────────

def load_eval_dataset(xlsx_path: Path = EVAL_XLSX_PATH) -> pd.DataFrame:
    """Load the Q&A evaluation dataset from the Excel file."""
    df = pd.read_excel(str(xlsx_path))
    print(f"✅ Loaded {len(df)} evaluation questions")
    print(f"   Columns: {df.columns.tolist()}")
    return df


# ── Run the pipeline on eval questions ────────────────────────────────────

def run_pipeline_on_eval_set(
    pipeline,               # RAGPipeline instance
    eval_df: pd.DataFrame,
    top_k: int = 5,
) -> list[dict]:
    """
    Run the RAG pipeline on every question in the eval dataset.

    Returns a list of dicts:
      - question          : str
      - answer            : str   (generated)
      - contexts          : list[str]  (retrieved chunks)
      - ground_truth      : str
      - ground_truth_context : str
      - page_number       : int
    """
    results = []

    for _, row in eval_df.iterrows():
        question = row["Question"]
        ground_truth = row["Ground_Truth_Answer"]
        gt_context   = row["Ground_Truth_Context"]
        page_num     = row.get("Page_Number", None)

        print(f"\n🔍 Q: {question[:80]}...")

        # Retrieve context
        chunks = pipeline.get_context(question, top_k=top_k)
        contexts = [c["text"] for c in chunks]

        # Generate answer directly from already-retrieved context to avoid
        # a second retrieval call (major latency saver in evaluation runs).
        answer = generate_answer(question, chunks)

        print(f"   A: {answer[:100]}...")

        results.append({
            "question"            : question,
            "answer"              : answer,
            "contexts"            : contexts,
            "ground_truth"        : ground_truth,
            "ground_truth_context": gt_context,
            "page_number"         : page_num,
        })

    return results


# ── RAGAS Evaluation ──────────────────────────────────────────────────────

def _get_ragas_vertex_models():
    """
    Build RAGAS-compatible LLM + embedding wrappers using Vertex AI.
    This avoids any OpenAI API key dependency.
    """
    init_vertex()
    lc_llm = ChatVertexAI(
        model=GEMINI_MODEL,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        temperature=0.0,
    )
    lc_embeddings = VertexAIEmbeddings(
        model=EMBEDDING_MODEL,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )
    return LangchainLLMWrapper(lc_llm), LangchainEmbeddingsWrapper(lc_embeddings)

def run_ragas_evaluation(pipeline_results: list[dict]) -> dict:
    """
    Run RAGAS metrics on the pipeline results.

    Returns a dict of metric_name → score.
    """

    # RAGAS expects a Hugging Face Dataset with these exact column names
    ragas_data = {
        "question"    : [r["question"] for r in pipeline_results],
        "answer"      : [r["answer"] for r in pipeline_results],
        "contexts"    : [r["contexts"] for r in pipeline_results],
        "ground_truth": [r["ground_truth"] for r in pipeline_results],
    }

    dataset = Dataset.from_dict(ragas_data)
    ragas_llm, ragas_embeddings = _get_ragas_vertex_models()

    print("\n🧮 Running RAGAS evaluation...")
    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        batch_size=4,
    )

    scores_df = scores.to_pandas()
    numeric_scores_df = scores_df.select_dtypes(include=["number"])
    score_dict = numeric_scores_df.mean(numeric_only=True).to_dict()
    print("\n📊 RAGAS Scores:")
    for metric, score in score_dict.items():
        print(f"   {metric:<25} {score:.4f}")

    return score_dict


# ── LLM as a Judge ────────────────────────────────────────────────────────

def llm_judge(
    question: str,
    generated_answer: str,
    ground_truth: str,
) -> dict:
    """
    Use Gemini to judge the quality of a generated answer.
    Returns a dict: {score: int (1-5), reasoning: str, verdict: str}
    """
    init_vertex()
    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)

    judge_prompt = f"""You are an expert evaluator assessing the quality of an AI-generated answer.

Question: {question}
Generated Answer: {generated_answer}
Ground Truth Answer: {ground_truth}

Score the generated answer from 1 to 5:
  5 = Perfect match with ground truth, fully correct
  4 = Mostly correct, minor differences
  3 = Partially correct, some important info missing
  2 = Mostly incorrect, significant errors
  1 = Completely wrong or irrelevant

Respond ONLY with this JSON (no markdown):
{{
  "score": <integer 1-5>,
  "verdict": "correct" | "partial" | "incorrect",
  "reasoning": "<one sentence explanation>"
}}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=judge_prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
    )
    text = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        return {"score": 0, "verdict": "error", "reasoning": text}


def run_llm_judge_evaluation(pipeline_results: list[dict]) -> pd.DataFrame:
    """
    Run LLM-as-a-Judge on all pipeline results.
    Returns a DataFrame with scores.
    """
    rows = []
    for r in pipeline_results:
        judgment = llm_judge(r["question"], r["answer"], r["ground_truth"])
        rows.append({
            "question"  : r["question"][:80],
            "score"     : judgment.get("score", 0),
            "verdict"   : judgment.get("verdict", ""),
            "reasoning" : judgment.get("reasoning", ""),
        })

    df = pd.DataFrame(rows)
    avg_score = df["score"].mean()
    print(f"\n⚖️  LLM Judge Average Score: {avg_score:.2f} / 5.0")
    print(f"   Verdict breakdown:\n{df['verdict'].value_counts().to_string()}")
    return df


# ── Save evaluation results ───────────────────────────────────────────────

def save_evaluation_results(
    ragas_scores: dict,
    judge_df: pd.DataFrame,
    pipeline_name: str = "phase1_faiss",
    out_dir: Path = None,
):
    """Save evaluation results to disk."""
    if out_dir is None:
        out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # RAGAS scores
    ragas_file = out_dir / f"eval_ragas_{pipeline_name}.json"
    with open(ragas_file, "w") as f:
        json.dump(ragas_scores, f, indent=2)

    # Judge results
    judge_file = out_dir / f"eval_judge_{pipeline_name}.csv"
    judge_df.to_csv(judge_file, index=False)

    print(f"\n💾 Saved RAGAS scores  → {ragas_file}")
    print(f"💾 Saved Judge results → {judge_file}")


if __name__ == "__main__":
    # Quick test – requires pipeline to be already built
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(retriever_mode="faiss").load()
    eval_df  = load_eval_dataset()

    results = run_pipeline_on_eval_set(pipeline, eval_df, top_k=5)
    ragas   = run_ragas_evaluation(results)
    judge   = run_llm_judge_evaluation(results)
    save_evaluation_results(ragas, judge, pipeline_name="phase1_faiss")
