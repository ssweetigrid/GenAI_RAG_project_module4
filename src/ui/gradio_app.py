"""
src/ui/gradio_app.py
─────────────────────
Alternative Gradio UI for the RAG pipeline.
Gradio is simpler to deploy and has built-in chat interface support.

Run with:
    python src/ui/gradio_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr

from src.rag_pipeline         import RAGPipeline
from src.ingestion.chunk_text import load_chunks


# ── Load pipeline once at startup ────────────────────────────────────────
print("Loading RAG pipeline...")
try:
    _chunks   = load_chunks()
    _pipeline = RAGPipeline(retriever_mode="faiss").load(chunks=_chunks)
    print("✅ Pipeline ready")
except Exception as e:
    _pipeline = None
    print(f"❌ Pipeline failed to load: {e}")


def rag_chat(message: str, history: list, top_k: int, retriever_mode: str) -> str:
    """
    Main chat function called by Gradio.
    Returns the generated answer as a string.
    """
    if _pipeline is None:
        return "❌ Pipeline not loaded. Run Phase 1 first."

    if not message.strip():
        return "Please enter a question."

    # Use the selected retriever mode
    pipeline = RAGPipeline(retriever_mode=retriever_mode).load(chunks=_chunks)

    # Get context for display
    chunks = pipeline.get_context(message, top_k=top_k)
    context_info = "\n".join(
        f"[Page {c['page_number']} | {c['content_type']}] {c['text'][:100]}..."
        for c in chunks
    )

    # Generate answer
    answer = pipeline.query(message, top_k=top_k)

    # Append context as a footnote
    full_response = f"{answer}\n\n---\n**Retrieved from:** {[c['page_number'] for c in chunks]}"
    return full_response


def get_context_display(message: str, top_k: int) -> str:
    """Return retrieved context chunks as formatted text."""
    if _pipeline is None or not message.strip():
        return ""
    chunks = _pipeline.get_context(message, top_k=top_k)
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"**Chunk {i}** | Page {c['page_number']} | "
            f"Type: {c['content_type']} | Score: {c.get('score', 0):.3f}\n"
            f"{c['text'][:300]}...\n"
        )
    return "\n---\n".join(lines)


# ── Build Gradio interface ────────────────────────────────────────────────
with gr.Blocks(title="IFC Annual Report RAG", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📄 IFC Annual Report 2024 – RAG System")
    gr.Markdown("Ask questions about IFC's 2024 financial report using AI-powered search.")

    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.ChatInterface(
                fn=rag_chat,
                additional_inputs=[
                    gr.Slider(1, 10, value=5, label="Top-K chunks", step=1),
                    gr.Dropdown(
                        choices=["faiss", "qdrant", "hybrid"],
                        value="faiss",
                        label="Retriever mode",
                    ),
                ],
                examples=[
                    ["What was IFC's net income for fiscal year 2024?"],
                    ["What is the total value of IFC's assets?"],
                    ["What are the key risk factors mentioned in the report?"],
                    ["How much did IFC commit to climate investments?"],
                ],
                title="",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📚 Context Explorer")
            context_query = gr.Textbox(label="Enter query to see retrieved context")
            context_topk  = gr.Slider(1, 10, value=3, label="Top-K", step=1)
            context_btn   = gr.Button("Show Context")
            context_out   = gr.Markdown(label="Retrieved chunks")

            context_btn.click(
                fn=get_context_display,
                inputs=[context_query, context_topk],
                outputs=context_out,
            )

    gr.Markdown("---")
    gr.Markdown("**Powered by:** Gemini 2.0 Flash · Vertex AI · FAISS · Qdrant · LangChain")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # Set True to get a public URL
    )
