"""
src/retrieval/multihop_retriever.py
─────────────────────────────────────
Phase 4, Task 2 – Multi-hop Retrieval

What is multi-hop retrieval?
─────────────────────────────
A normal (single-hop) RAG pipeline does this:
    Question → retrieve chunks → generate answer

But some questions CANNOT be answered in one retrieval step because:
  • They require connecting information from different parts of the document
  • The answer to sub-question A is needed to know what to look for in sub-question B
  • The question spans multiple topics or time periods

Example of a multi-hop question:
  "How did IFC's net income change between 2023 and 2024, and what risk factors
   contributed to this change?"
                ↓
  Hop 1: Retrieve net income figures for 2023 and 2024
  Hop 2: Using those figures, retrieve risk factor sections relevant to the change
  Hop 3: Synthesise both sets of context into one final answer

How this implementation works:
────────────────────────────────
We use Gemini itself as the "query decomposer" – it breaks a complex question
into simpler sub-questions, we answer each one, then Gemini synthesises the
final answer from all the intermediate answers.

                ┌─────────────────────────────────────┐
                │  Complex question                   │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Gemini: decompose into              │
                │  N sub-questions (JSON output)       │
                └──────────────┬──────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Sub-Q 1                Sub-Q 2  ...
                    │                     │
             Retrieve chunks       Retrieve chunks
                    │                     │
             Generate partial      Generate partial
             answer 1              answer 2
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Gemini: synthesise all partial      │
                │  answers into one final answer       │
                └─────────────────────────────────────┘
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from config.settings import (
    GEMINI_MODEL, GCP_PROJECT_ID, GCP_LOCATION,
    TOP_K,
)
from config.gcp_auth import init_vertex


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class Hop:
    """
    Represents one retrieval hop.

    Attributes
    ----------
    sub_question    : the refined query used in this hop
    retrieved_chunks: list of context dicts from the vector store
    partial_answer  : Gemini's answer to this sub-question only
    hop_number      : which hop this is (1-based)
    """
    sub_question     : str
    retrieved_chunks : list[dict] = field(default_factory=list)
    partial_answer   : str        = ""
    hop_number       : int        = 1


@dataclass
class MultiHopResult:
    """
    Final result from a multi-hop retrieval run.

    Attributes
    ----------
    original_question : the user's original complex question
    sub_questions     : list of sub-questions Gemini generated
    hops              : list of Hop objects (one per sub-question)
    final_answer      : synthesised answer across all hops
    total_hops        : how many retrieval hops were done
    """
    original_question : str
    sub_questions     : list[str]
    hops              : list[Hop]
    final_answer      : str
    total_hops        : int


# ── Gemini helpers ────────────────────────────────────────────────────────

def _get_client():
    """Get a Gemini client (initialises Vertex AI if needed)."""
    init_vertex()
    from google import genai
    return genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)


def _get_langfuse():
    """Return a Langfuse client if credentials are configured, else None."""
    from config.settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    if not LANGFUSE_PUBLIC_KEY:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception:
        return None


# ── Step 1: Decompose the question into sub-questions ─────────────────────

def decompose_question(question: str, max_hops: int = 3) -> list[str]:
    """
    Use Gemini to break a complex question into simpler sub-questions.

    Each sub-question should be answerable independently from the document.
    Returns a list of sub-question strings (length 1 to max_hops).

    Example
    -------
    Input:
        "How did IFC net income change from 2023 to 2024, and what
         were the main contributing risk factors?"

    Output:
        [
          "What was IFC net income in fiscal year 2023?",
          "What was IFC net income in fiscal year 2024?",
          "What risk factors does IFC identify as affecting net income?"
        ]
    """
    client = _get_client()
    from google.genai import types

    decompose_prompt = f"""You are a question analyst for a financial document RAG system.

A user asked this complex question about the IFC 2024 Annual Report:
"{question}"

Break this into {max_hops} or fewer SIMPLE sub-questions that together cover the full answer.
Each sub-question must be:
  1. Self-contained and searchable in the document
  2. Simpler than the original question
  3. Answerable from financial report text

If the question is already simple enough to answer in one retrieval step,
return just one sub-question (the original).

Respond ONLY with a JSON array of strings, no markdown, no explanation:
["sub-question 1", "sub-question 2", ...]"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=decompose_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,          # deterministic decomposition
            max_output_tokens=512,
        ),
    )

    text = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        sub_questions = json.loads(text)
        # Validate: must be a list of strings
        if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
            # Cap at max_hops
            sub_questions = sub_questions[:max_hops]
            print(f"  ✅ Decomposed into {len(sub_questions)} sub-questions:")
            for i, sq in enumerate(sub_questions, 1):
                print(f"     Hop {i}: {sq}")
            return sub_questions
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: treat the original as a single-hop question
    print("  ⚠️  Decomposition failed, falling back to single hop")
    return [question]


# ── Step 2: Answer each sub-question individually ─────────────────────────

def answer_sub_question(
    sub_question : str,
    pipeline,
    hop_number   : int = 1,
    top_k        : int = TOP_K,
    previous_hops: list[Hop] = None,
) -> Hop:
    """
    Retrieve context and generate a partial answer for one sub-question.

    If previous_hops is provided, the partial answers from earlier hops
    are injected into the prompt so Gemini can use them as extra context.
    This is the key "iterative refinement" mechanism of multi-hop RAG.

    Parameters
    ----------
    sub_question  : the sub-question for this hop
    pipeline      : RAGPipeline instance
    hop_number    : 1-based index of this hop
    top_k         : chunks to retrieve
    previous_hops : list of Hop objects already completed (may be empty)

    Returns
    -------
    Hop object with retrieved_chunks and partial_answer filled in
    """
    print(f"\n  🔍 Hop {hop_number}: '{sub_question[:70]}...'")

    # Retrieve chunks from the vector store
    retrieved_chunks = pipeline.get_context(sub_question, top_k=top_k)
    print(f"     Retrieved {len(retrieved_chunks)} chunks from pages "
          f"{[c['page_number'] for c in retrieved_chunks]}")

    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Page {chunk['page_number']} | "
            f"Type: {chunk.get('content_type', 'text')}]\n{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    # Inject answers from previous hops as additional context
    prior_knowledge = ""
    if previous_hops:
        prior_parts = []
        for prev in previous_hops:
            prior_parts.append(
                f"Sub-question {prev.hop_number}: {prev.sub_question}\n"
                f"Answer: {prev.partial_answer}"
            )
        prior_knowledge = (
            "\n\n=== ANSWERS FROM PREVIOUS HOPS ===\n"
            + "\n\n".join(prior_parts)
            + "\n\n(Use the above as additional context when answering below)"
        )

    # Prompt for this hop
    prompt = f"""You are a financial analyst assistant for the IFC 2024 Annual Report.

Answer ONLY the specific sub-question below using the provided context.
Be concise and factual. Cite page numbers.
{prior_knowledge}

=== RETRIEVED CONTEXT ===
{context_str}

=== SUB-QUESTION (Hop {hop_number}) ===
{sub_question}

=== PARTIAL ANSWER ==="""

    client = _get_client()
    from google.genai import types
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        ),
    )
    partial_answer = response.text.strip()
    print(f"     Answer preview: {partial_answer[:100]}...")

    return Hop(
        sub_question     = sub_question,
        retrieved_chunks = retrieved_chunks,
        partial_answer   = partial_answer,
        hop_number       = hop_number,
    )


# ── Step 3: Synthesise all partial answers into one final answer ──────────

def synthesise_final_answer(
    original_question : str,
    hops              : list[Hop],
) -> str:
    """
    Ask Gemini to combine all partial answers into one coherent final answer.

    This step ensures:
      • The final answer directly addresses the original (complex) question
      • Information from all hops is woven together
      • Contradictions or overlaps are resolved
      • Source pages are cited
    """
    print(f"\n  🧠 Synthesising final answer from {len(hops)} hops...")

    # Format all hop results
    hop_summaries = []
    all_pages = []
    for hop in hops:
        hop_summaries.append(
            f"Sub-question {hop.hop_number}: {hop.sub_question}\n"
            f"Partial answer: {hop.partial_answer}\n"
            f"Source pages: {[c['page_number'] for c in hop.retrieved_chunks]}"
        )
        all_pages.extend(c['page_number'] for c in hop.retrieved_chunks)

    hops_text = "\n\n".join(hop_summaries)
    unique_pages = sorted(set(all_pages))

    synthesis_prompt = f"""You are a financial analyst synthesising research notes.

The following sub-questions were answered step by step to address a complex question
about the IFC 2024 Annual Report. Combine all the partial answers into ONE
comprehensive, well-structured final answer that directly addresses the original question.

=== ORIGINAL QUESTION ===
{original_question}

=== PARTIAL ANSWERS FROM EACH HOP ===
{hops_text}

=== INSTRUCTIONS ===
- Write a single coherent answer (not bullet points of partial answers)
- Integrate the information naturally
- Cite specific page numbers (pages used: {unique_pages})
- If the partial answers contain any contradictions, note them

=== FINAL ANSWER ==="""

    client = _get_client()
    from google.genai import types
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


# ── Main multi-hop pipeline ───────────────────────────────────────────────

class MultiHopRetriever:
    """
    Orchestrates the full multi-hop retrieval process.

    Usage
    -----
    retriever = MultiHopRetriever(pipeline, max_hops=3)
    result    = retriever.run("How did IFC net income change and why?")

    print(result.final_answer)
    print(f"Used {result.total_hops} hops")
    for hop in result.hops:
        print(f"  Hop {hop.hop_number}: {hop.sub_question}")
    """

    def __init__(self, pipeline, max_hops: int = 3, top_k: int = TOP_K):
        """
        Parameters
        ----------
        pipeline  : RAGPipeline instance (must already have .load() called)
        max_hops  : maximum number of retrieval hops allowed
        top_k     : chunks to retrieve per hop
        """
        self.pipeline  = pipeline
        self.max_hops  = max_hops
        self.top_k     = top_k
        self._langfuse = _get_langfuse()

    def run(self, question: str, session_id: str = "multihop") -> MultiHopResult:
        """
        Run the full multi-hop retrieval pipeline for a complex question.

        Returns a MultiHopResult containing the final answer and all
        intermediate hop details for transparency.
        """
        print(f"\n{'='*60}")
        print(f"🔗 Multi-hop RAG")
        print(f"   Question: {question[:80]}...")
        print(f"   Max hops: {self.max_hops}")
        print("=" * 60)

        # Start Langfuse trace
        trace = None
        if self._langfuse:
            trace = self._langfuse.trace(
                name="multihop-rag",
                session_id=session_id,
                input={"question": question, "max_hops": self.max_hops},
            )

        # ── Step 1: Decompose ────────────────────────────────────────────
        print("\n📋 Step 1: Decomposing question into sub-questions...")
        sub_questions = decompose_question(question, max_hops=self.max_hops)

        # ── Step 2: Answer each sub-question (with iterative context) ────
        print("\n🔄 Step 2: Running retrieval hops...")
        hops: list[Hop] = []

        for i, sub_q in enumerate(sub_questions, 1):
            hop = answer_sub_question(
                sub_question  = sub_q,
                pipeline      = self.pipeline,
                hop_number    = i,
                top_k         = self.top_k,
                previous_hops = hops,          # pass completed hops as context
            )
            hops.append(hop)

        # ── Step 3: Synthesise final answer ──────────────────────────────
        print("\n✨ Step 3: Synthesising final answer...")
        if len(hops) == 1:
            # Only one hop → no need to synthesise, use the partial answer
            final_answer = hops[0].partial_answer
            print("  (Single hop – using partial answer directly)")
        else:
            final_answer = synthesise_final_answer(question, hops)

        # Log to Langfuse
        if trace:
            self._langfuse.trace(
                name="multihop-rag",
                session_id=session_id,
                output={
                    "final_answer": final_answer,
                    "total_hops"  : len(hops),
                    "sub_questions": sub_questions,
                },
            )
            self._langfuse.flush()

        result = MultiHopResult(
            original_question = question,
            sub_questions     = sub_questions,
            hops              = hops,
            final_answer      = final_answer,
            total_hops        = len(hops),
        )

        print(f"\n✅ Multi-hop complete  |  {result.total_hops} hop(s)")
        return result

    def run_single_hop(self, question: str) -> str:
        """
        Shortcut: skip decomposition and run exactly one hop.
        Useful for comparing single-hop vs multi-hop answers side-by-side.
        """
        chunks = self.pipeline.get_context(question, top_k=self.top_k)
        return self.pipeline.query(question, top_k=self.top_k)


# ── Utility: is this question complex enough for multi-hop? ───────────────

def needs_multihop(question: str) -> bool:
    """
    Heuristic check: does this question likely need multi-hop retrieval?

    Looks for keywords that suggest the question spans multiple concepts
    or requires sequential information gathering.

    Returns True if multi-hop is recommended.
    """
    multihop_indicators = [
        # Comparison / change indicators
        "compared to", "change", "difference", "versus", "vs",
        "increased", "decreased", "grew", "fell", "between",
        # Causal / reasoning indicators
        "why", "because", "reason", "cause", "result", "led to",
        "contributed", "impact", "effect", "due to",
        # Multi-part indicators
        "and also", "as well as", "in addition", "furthermore",
        "both", "all", "each", "across",
        # Time span indicators
        "over the years", "trend", "historically", "from 20", "to 20",
        "year-over-year", "annual", "quarterly",
        # Relationship indicators
        "how does", "relationship between", "correlation",
        "how did", "what caused",
    ]
    q_lower = question.lower()
    matched = [kw for kw in multihop_indicators if kw in q_lower]

    if matched:
        print(f"  💡 Multi-hop recommended  (matched: {matched[:3]})")
        return True

    print("  💡 Single-hop likely sufficient")
    return False
