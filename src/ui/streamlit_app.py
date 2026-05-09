"""
src/ui/streamlit_app.py
────────────────────────
Streamlit UI for IFC RAG system (Phases 1–6).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import ROOT_DIR
from src.embeddings.faiss_store import load_faiss_index, search_faiss
from src.evaluation.compare_phases import compare_all_phases
from src.evaluation.ragas_eval import (
    load_eval_dataset,
    run_pipeline_on_eval_set,
    run_ragas_evaluation,
)
from src.rag_pipeline import RAGPipeline
from src.retrieval.multihop_retriever import MultiHopRetriever
from src.retrieval.retriever import embed_query
from src.retrieval.semantic_cache import SemanticCache


for key, default in [
    ("query", ""),
    ("cache_hits", 0),
    ("cache_misses", 0),
    ("cache_entries", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


st.set_page_config(page_title="IFC RAG", page_icon="📄", layout="wide")

st.markdown(
    """<style>
.stApp { background-color: #0a0e17 !important; color: #e8edf5; }
section[data-testid="stSidebar"] { background-color: #0f1520 !important; }
.stButton>button { background: #1d6aff !important; color: white !important;
  border: none !important; border-radius: 6px !important; }
.stTabs [data-baseweb="tab"] { color: #8b9ab5; }
.stTabs [aria-selected="true"] { color: #e8edf5 !important; }
</style>""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 📄 IFC RAG")
    st.caption("Annual Report 2024 · Phases 1–6")
    st.divider()

    selected_phase = st.selectbox(
        "Workflow Phase",
        [
            "Phase 1 — Retrieval",
            "Phase 2 — Evaluation",
            "Phase 3 — Hybrid + Re-rank",
            "Phase 4 — Cache + Multi-hop",
            "Phase 5 — Multimodal",
            "Phase 6 — ColPali",
        ],
        help="Choose the phase you want to work in. Settings adapt automatically.",
    )

    top_k = st.slider("Top-K Chunks", 1, 15, 5)
    retriever_mode = "faiss"
    use_rerank = False
    use_cache = False
    use_multihop = False
    max_hops = 3
    content_type = "All"
    use_page_filter = False

    if selected_phase == "Phase 1 — Retrieval":
        retriever_mode = st.selectbox("Vector Store", ["faiss", "qdrant", "hybrid"])
    elif selected_phase == "Phase 2 — Evaluation":
        retriever_mode = st.selectbox("Vector Store", ["faiss", "qdrant", "hybrid"])
    elif selected_phase == "Phase 3 — Hybrid + Re-rank":
        retriever_mode = "hybrid"
        use_rerank = True
        st.caption("Auto: `hybrid` retrieval + rerank enabled")
        content_type = st.selectbox("Content Type", ["All", "text", "table", "image"], index=0)
        use_page_filter = st.checkbox("Page Range Filter", value=False)
    elif selected_phase == "Phase 4 — Cache + Multi-hop":
        retriever_mode = st.selectbox("Vector Store", ["faiss", "qdrant", "hybrid"], index=2)
        use_cache = st.checkbox("Semantic Cache", value=True)
        use_multihop = st.checkbox("Multi-hop Retrieval", value=True)
        max_hops = st.slider("Max Hops", 1, 4, 3) if use_multihop else 3
    elif selected_phase == "Phase 5 — Multimodal":
        retriever_mode = st.selectbox("Vector Store", ["faiss", "qdrant", "hybrid"], index=0)
        content_type = st.selectbox("Content Type", ["All", "text", "table", "image"], index=0)
        use_page_filter = st.checkbox("Page Range Filter", value=False)
    elif selected_phase == "Phase 6 — ColPali":
        retriever_mode = "faiss"
        st.caption("Use the ColPali tab for visual search")

    page_min, page_max = 1, 100
    if use_page_filter:
        c1, c2 = st.columns(2)
        page_min = c1.number_input("From", 1, 200, 1)
        page_max = c2.number_input("To", 1, 200, 100)
    structured = st.checkbox("Structured JSON Answer")
    st.divider()
    st.caption(
        f"Phase: {selected_phase} | mode={retriever_mode} | top_k={top_k}"
        + (" | rerank=on" if use_rerank else "")
    )


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline(mode, rerank, struct, phase):
    from src.ingestion.chunk_text import load_chunks

    if phase == "Phase 5 — Multimodal":
        from src.retrieval.multimodal_retriever import load_all_chunks_multimodal
        chunks = load_all_chunks_multimodal()
    else:
        chunks = load_chunks()
    pipeline = RAGPipeline(retriever_mode=mode, use_rerank=rerank, structured=struct).load(
        chunks=chunks
    )
    return pipeline, chunks


@st.cache_resource
def load_cache_obj():
    return SemanticCache(threshold=0.92)


try:
    pipeline, chunks = load_pipeline(retriever_mode, use_rerank, structured, selected_phase)
    cache_obj = load_cache_obj()
    st.success("✅ Pipeline ready")
except Exception as e:
    st.error(f"❌ Pipeline failed: {e}")
    st.code("python scripts/run_phase1.py", language="bash")
    st.stop()


st.title("📄 IFC Annual Report 2024 – RAG System")
st.caption("Multimodal Retrieval-Augmented Generation · Phases 1–6")

tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat", "⚡ FAISS vs Qdrant", "📊 Evaluation", "🌿 ColPali (Phase 6)"]
)

with tab1:
    sample_questions = [
        "What was IFC's net income for fiscal year 2024?",
        "How much did IFC commit to climate-related investments?",
        "What are key risk factors in the report?",
        "How did 2023 vs 2024 financial performance change?",
        "What is IFC's total assets figure?",
    ]
    sample_cols = st.columns(5)
    for i, q in enumerate(sample_questions):
        if sample_cols[i].button(q[:32] + ("..." if len(q) > 32 else ""), help=q, use_container_width=True):
            st.session_state.query = q

    query = st.text_input(
        "Your question:",
        value=st.session_state.get("query", ""),
        placeholder="Ask anything about IFC 2024 Annual Report...",
    )
    ask_clicked = st.button("🔍 Ask", type="primary")

    if ask_clicked and query:
        if use_multihop:
            st.info("🔗 Multi-hop mode — decomposing question into sub-queries...")
            mh = MultiHopRetriever(pipeline, max_hops=max_hops, top_k=top_k)
            result = mh.run(query)
            with st.expander(f"🔗 Hop Details ({result.total_hops} hops)"):
                for hop in result.hops:
                    st.markdown(f"**Hop {hop.hop_number}:** {hop.sub_question}")
                    st.caption(f"Pages: {[c['page_number'] for c in hop.retrieved_chunks]}")
                    st.text(hop.partial_answer[:200])
                    st.divider()
            st.subheader("💡 Final Answer")
            st.markdown(result.final_answer)
        else:
            if use_cache:
                cached_ans, sim = cache_obj.lookup(query)
                if cached_ans:
                    st.success(f"⚡ Cache hit  (similarity={sim:.3f}) — returned instantly")
                    st.session_state.cache_hits = st.session_state.get("cache_hits", 0) + 1
                    st.subheader("💡 Answer *(from cache)*")
                    st.markdown(cached_ans)
                    st.stop()
                st.session_state.cache_misses = st.session_state.get("cache_misses", 0) + 1

            ctx_chunks = pipeline.get_context(
                query,
                top_k=top_k,
                page_range=(int(page_min), int(page_max)) if use_page_filter else None,
                content_type=content_type if content_type != "All" else None,
            )

            with st.expander(f"📚 Retrieved Context ({len(ctx_chunks)} chunks)"):
                for i, c in enumerate(ctx_chunks, 1):
                    score = c.get("rerank_score", c.get("score", 0))
                    st.markdown(
                        f"**Chunk {i}** · Page `{c['page_number']}` · "
                        f"Type `{c['content_type']}` · Score `{score:.3f}`"
                    )
                    st.text(c["text"][:300])
                    st.divider()

            st.subheader("💡 Answer")
            full_answer = ""
            if structured:
                result = pipeline.query(
                    query,
                    top_k=top_k,
                    page_range=(int(page_min), int(page_max)) if use_page_filter else None,
                    content_type=content_type if content_type != "All" else None,
                )
                if isinstance(result, dict):
                    st.json(result)
                    full_answer = result.get("answer", "")
                else:
                    st.markdown(result)
                    full_answer = str(result)
            else:
                answer_placeholder = st.empty()
                for token in pipeline.query(
                    query,
                    top_k=top_k,
                    page_range=(int(page_min), int(page_max)) if use_page_filter else None,
                    content_type=content_type if content_type != "All" else None,
                    stream=True,
                ):
                    full_answer += token
                    answer_placeholder.markdown(full_answer)

            if use_cache and full_answer:
                cache_obj.store(query, full_answer, context_chunks=ctx_chunks)
                st.session_state.cache_entries = len(cache_obj.entries)
                st.caption("💾 Stored in semantic cache")

            pages_used = sorted({c["page_number"] for c in ctx_chunks})
            st.caption(f"📎 Sources: pages {pages_used}")

with tab2:
    st.markdown("Run the same query through FAISS and Qdrant and compare latency.")
    test_query = st.text_input(
        "Test query", value="What was IFC's net income for fiscal year 2024?"
    )
    if st.button("Run side-by-side retrieval"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚡ FAISS")
            import time

            t0 = time.perf_counter()
            index, meta = load_faiss_index()
            q_emb = embed_query(test_query)
            faiss_results = search_faiss(q_emb, index, meta, top_k=top_k)
            faiss_time = (time.perf_counter() - t0) * 1000
            st.metric("Latency", f"{faiss_time:.1f} ms")
            st.write("Pages:", [r["page_number"] for r in faiss_results])
            st.text(faiss_results[0]["text"][:200] if faiss_results else "No results")

        with col2:
            st.markdown("### 🗃️ Qdrant")
            try:
                from src.embeddings.qdrant_store import get_qdrant_client, search_qdrant

                t0 = time.perf_counter()
                qclient = get_qdrant_client()
                qdrant_results = search_qdrant(q_emb, qclient, top_k=top_k)
                qdrant_time = (time.perf_counter() - t0) * 1000
                st.metric("Latency", f"{qdrant_time:.1f} ms")
                st.write("Pages:", [r["page_number"] for r in qdrant_results])
                st.text(qdrant_results[0]["text"][:200] if qdrant_results else "No results")
            except Exception:
                st.error("Qdrant not available. Run: cd docker && docker-compose up -d")

with tab3:
    st.markdown("Evaluate pipeline quality using RAGAS metrics and LLM-as-a-Judge.")
    col1, col2 = st.columns(2)
    if col1.button("📊 Run RAGAS Evaluation"):
        with st.spinner("Running RAGAS on evaluation dataset..."):
            eval_df = load_eval_dataset()
            results = run_pipeline_on_eval_set(pipeline, eval_df.head(10), top_k=top_k)
            try:
                ragas_scores = run_ragas_evaluation(results)
            except Exception as e:
                ragas_scores = None
                st.warning(
                    "RAGAS is unavailable in this environment (missing provider credentials). "
                    "LLM-as-a-Judge continues to work.\n\n"
                    f"Details: {e}"
                )
        if ragas_scores:
            st.success("RAGAS evaluation complete!")
            st.json(ragas_scores)
            scores_df = pd.DataFrame([ragas_scores])
            st.bar_chart(scores_df)
            compare_all_phases()

    if col2.button("⚖️ Run LLM-as-Judge"):
        with st.spinner("Running LLM Judge..."):
            eval_df = load_eval_dataset()
            results = run_pipeline_on_eval_set(pipeline, eval_df.head(5), top_k=top_k)
            from src.evaluation.ragas_eval import run_llm_judge_evaluation

            judge_df = run_llm_judge_evaluation(results)
        st.success("Judge evaluation complete!")
        st.dataframe(judge_df)
        avg = judge_df["score"].mean()
        st.metric("Average Score", f"{avg:.2f} / 5.0")

with tab4:
    visual_query = st.text_input("Visual query")
    if st.button("🔍 Search by Visual Content"):
        from src.retrieval.colpali_retriever import answer_with_visual_context

        try:
            colpali_index_path = str(ROOT_DIR / "data/processed/faiss_colpali")
            index, meta = load_faiss_index(index_path=colpali_index_path)
            q_emb = embed_query(visual_query)
            top_pages = search_faiss(q_emb, index, meta, top_k=3)
            cols = st.columns(len(top_pages)) if top_pages else []
            for col, page in zip(cols, top_pages):
                img_path = page.get("image_path", "")
                if img_path and Path(img_path).exists():
                    col.image(img_path, caption=f"Page {page['page_number']}", use_container_width=True)
            st.subheader("💡 Visual Answer")
            answer = answer_with_visual_context(visual_query, top_pages)
            st.markdown(answer)
        except FileNotFoundError:
            st.error("ColPali index not found. Run scripts/run_phase6_colpali.py first.")
