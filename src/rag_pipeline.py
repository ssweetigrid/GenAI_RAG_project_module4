"""
src/rag_pipeline.py
────────────────────
The main RAG pipeline that wires everything together.

Phases supported:
  Phase 1  – Text-only RAG
  Phase 3  – Hybrid + re-ranking
  Phase 5  – Multimodal (text + tables + images)
"""

from __future__ import annotations
from typing import Literal

from src.retrieval.retriever import Retriever
from src.generation.generator import (
    generate_answer,
    generate_answer_streaming,
    generate_structured_answer,
)


class RAGPipeline:
    """
    One-stop class for the entire RAG workflow.

    Parameters
    ----------
    retriever_mode : "faiss" | "qdrant" | "hybrid"
    use_rerank     : bool   – enable cross-encoder re-ranking (Phase 3)
    structured     : bool   – return structured JSON instead of plain text
    """

    def __init__(
        self,
        retriever_mode: Literal["faiss", "qdrant", "hybrid"] = "faiss",
        use_rerank: bool = False,
        structured: bool = False,
    ):
        self.retriever_mode = retriever_mode
        self.use_rerank     = use_rerank
        self.structured     = structured
        self.retriever      = Retriever(mode=retriever_mode)

    def load(self, chunks: list[dict] = None):
        """Load the vector stores. Call this once before querying."""
        self.retriever.load(chunks=chunks)
        return self

    def query(
        self,
        question: str,
        top_k: int = 5,
        page_range: tuple = None,
        content_type: str = None,
        session_id: str = "default",
        stream: bool = False,
    ):
        """
        Run the full RAG pipeline for a user question.

        Returns
        -------
        If stream=True  → a generator that yields text tokens
        If structured   → a dict  {answer, sources, confidence}
        Else            → a plain string answer
        """

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_rerank=self.use_rerank,
            page_range=page_range,
            content_type=content_type,
        )

        # Step 2: Generate answer
        if stream:
            return generate_answer_streaming(question, chunks, session_id)
        elif self.structured:
            return generate_structured_answer(question, chunks)
        else:
            return generate_answer(question, chunks, session_id)

    def get_context(
        self,
        question: str,
        top_k: int = 5,
        page_range: tuple = None,
        content_type: str = None,
    ) -> list[dict]:
        """Return just the retrieved context (no generation). Useful for evaluation."""
        return self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_rerank=self.use_rerank,
            page_range=page_range,
            content_type=content_type,
        )
