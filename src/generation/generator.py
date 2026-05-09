"""
src/generation/generator.py
────────────────────────────
Phase 1, Task 4 – Generation
Send retrieved context + user query to Gemini 2.0 Flash.

Features implemented here:
  • Streaming output (prints tokens as they arrive)
  • Structured / JSON output mode
  • Source citation in the answer
  • Langfuse tracing (observability)
"""

from __future__ import annotations

import json
from typing import Generator

from google import genai
from google.genai import types
from langfuse import Langfuse

from config.settings import (
    GEMINI_MODEL, GCP_PROJECT_ID, GCP_LOCATION,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)
from config.gcp_auth import init_vertex


# ── Lazy singletons ───────────────────────────────────────────────────────
_genai_client   = None
_langfuse_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        init_vertex()
        _genai_client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
        )
    return _genai_client


def _get_langfuse():
    global _langfuse_client
    if _langfuse_client is None and LANGFUSE_PUBLIC_KEY:
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return _langfuse_client


# ── Prompt builder ────────────────────────────────────────────────────────

def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the prompt that combines retrieved context with the user query.
    Each chunk is labelled with its page number for source attribution.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        page = chunk.get("page_number", "?")
        ctype = chunk.get("content_type", "text")
        context_parts.append(
            f"[Context {i} | Page {page} | Type: {ctype}]\n{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a financial analyst assistant specialising in IFC (International Finance Corporation) reports.

Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I could not find this information in the provided context."
Always cite which page number your answer comes from.

=== CONTEXT ===
{context_str}

=== QUESTION ===
{query}

=== ANSWER ==="""

    return prompt


# ── Streaming answer generation ───────────────────────────────────────────

def generate_answer_streaming(
    query: str,
    context_chunks: list[dict],
    session_id: str = "default",
) -> Generator[str, None, None]:
    """
    Generate a RAG answer using Gemini with streaming.
    Yields tokens one-by-one as they arrive from the API.

    Usage:
        for token in generate_answer_streaming(query, chunks):
            print(token, end="", flush=True)
    """
    client  = _get_genai_client()
    langfuse = _get_langfuse()
    prompt  = build_rag_prompt(query, context_chunks)

    # Start a Langfuse trace for observability
    trace = None
    if langfuse:
        trace = langfuse.trace(
            name="rag-query",
            session_id=session_id,
            input={"query": query, "num_chunks": len(context_chunks)},
        )

    full_response = []

    try:
        # Stream the response from Gemini
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,        # Low temperature for factual answers
                max_output_tokens=1024,
            ),
        ):
            token = chunk.text or ""
            full_response.append(token)
            yield token

    finally:
        # Log the complete response to Langfuse
        if trace:
            trace.update(output={"response": "".join(full_response)})
            langfuse.flush()


def generate_answer(
    query: str,
    context_chunks: list[dict],
    session_id: str = "default",
) -> str:
    """
    Non-streaming version of generate_answer.
    Returns the complete answer as a string.
    """
    return "".join(generate_answer_streaming(query, context_chunks, session_id))


# ── Structured / JSON output ──────────────────────────────────────────────

def generate_structured_answer(
    query: str,
    context_chunks: list[dict],
) -> dict:
    """
    Ask Gemini to return a structured JSON response with:
      - answer  : str
      - sources : list[int]  (page numbers used)
      - confidence : str     (high / medium / low)
    """
    client = _get_genai_client()
    prompt = build_rag_prompt(query, context_chunks)
    prompt += """

Respond ONLY with a JSON object (no markdown, no extra text) in this format:
{
  "answer": "...",
  "sources": [page numbers as integers],
  "confidence": "high" | "medium" | "low"
}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        ),
    )

    try:
        # Parse the JSON
        text = response.text.strip()
        # Remove markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": response.text, "sources": [], "confidence": "low"}
