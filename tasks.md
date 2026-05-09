# 🗂️ RAG Project – Complete Task Flow & Explanation

> **Dataset**: IFC Annual Report 2024 Financials (PDF) + RAG Evaluation Dataset (XLSX)
> **LLM**: Gemini 2.0 Flash via Vertex AI (no API keys – uses `sweeti.json`)
> **Tech Stack**: LangChain · FAISS · Qdrant · Streamlit · Langfuse · RAGAS · Docker

---

## 📁 Project Folder Structure

```
rag_project/
│
├── keys/
│   └── sweeti.json                  ← PUT YOUR GCP KEY HERE
│
├── data/
│   ├── raw/
│   │   ├── ifc-annual-report-2024-financials.pdf
│   │   └── RAG_evaluation_dataset.xlsx
│   └── processed/
│       ├── text/
│       │   ├── pages_text.json      ← extracted text per page
│       │   ├── chunks.json          ← overlapping text chunks
│       │   └── embeddings.json      ← 768-dim vectors per chunk
│       ├── tables/
│       │   ├── tables.json          ← all tables as markdown
│       │   └── tables_markdown/     ← one .md file per table
│       ├── images/
│       │   ├── images_metadata.json ← captions + metadata
│       │   ├── pages/               ← page raster images (Phase 6)
│       │   └── page_descriptions.json
│       ├── faiss_index.faiss        ← FAISS binary index
│       ├── faiss_index.pkl          ← chunk metadata
│       └── semantic_cache.json      ← Phase 4 cache
│
├── src/
│   ├── ingestion/
│   │   ├── extract_text.py          ← Phase 1: PDF text extraction
│   │   ├── extract_tables.py        ← Phase 5: Table extraction
│   │   ├── extract_images.py        ← Phase 5: Image + Gemini caption
│   │   └── chunk_text.py            ← Phase 1: Text chunking
│   ├── embeddings/
│   │   ├── embed.py                 ← Phase 1: Vertex AI embeddings
│   │   ├── faiss_store.py           ← Phase 1: FAISS index
│   │   └── qdrant_store.py          ← Phase 1: Qdrant collection
│   ├── retrieval/
│   │   ├── retriever.py             ← Phase 1+3: Unified retriever
│   │   ├── multimodal_retriever.py  ← Phase 5: Text+Table+Image chunks
│   │   ├── colpali_retriever.py     ← Phase 6: Page-image retrieval
│   │   └── semantic_cache.py        ← Phase 4: Semantic cache
│   ├── generation/
│   │   └── generator.py             ← Phase 1: Gemini generation
│   ├── evaluation/
│   │   └── ragas_eval.py            ← Phase 2: RAGAS + LLM Judge
│   ├── ui/
│   │   └── streamlit_app.py         ← Phase 1: Streamlit UI
│   └── rag_pipeline.py              ← Main pipeline orchestrator
│
├── scripts/
│   ├── run_phase1.py                ← Run Phase 1 end-to-end
│   ├── run_phase2_eval.py           ← Run evaluation
│   ├── run_phase3_hybrid.py         ← Run hybrid + re-ranking
│   ├── run_phase5_multimodal.py     ← Run multimodal ingestion
│   └── run_phase6_colpali.py        ← Run ColPali visual RAG
│
├── config/
│   ├── settings.py                  ← All env variables in one place
│   └── gcp_auth.py                  ← Vertex AI authentication
│
├── docker/
│   └── docker-compose.yml           ← Qdrant + Langfuse containers
│
├── Dockerfile                       ← Container for the RAG app
├── requirements.txt
└── .env.example                     ← Copy to .env and fill in values
```

---

## ⚙️ One-Time Setup (Do This First!)

### 1. Copy your GCP key
```bash
# Create the keys folder and put your JSON key there
mkdir -p keys
cp /path/to/your/sweeti.json keys/sweeti.json
```

### 2. Create your `.env` file
```bash
cp .env.example .env
# Now open .env and set your GCP_PROJECT_ID
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Docker services (Qdrant + Langfuse)
```bash
cd docker
docker-compose up -d
# Qdrant UI  → http://localhost:6333/dashboard
# Langfuse   → http://localhost:3000
```

---

## 🗺️ Phase-by-Phase Walkthrough

---

### 📌 Data Parsing (Pre-requisite for all phases)

**What happens:**
- `extract_text.py` reads each PDF page using **pdfplumber** and saves the raw text with page-level metadata (page number, source file, document title/author from pypdf)
- `extract_tables.py` uses **pdfplumber**'s table detector to find every table in the PDF, converts them to Markdown format (human-readable) and saves them as JSON
- `extract_images.py` uses **PyMuPDF (fitz)** to extract embedded images, filters out tiny icons, and sends each image to **Gemini 2.0 Flash** with a detailed prompt asking for a descriptive caption

**Files produced:**
- `data/processed/text/pages_text.json`
- `data/processed/tables/tables.json`
- `data/processed/images/images_metadata.json`

---

### 📌 Phase 1 – Text-Based RAG (Naive RAG)

**Goal:** Build a working baseline RAG that can answer questions using only the PDF's text.

**Steps:**

#### Step 1a: Text Extraction
- `src/ingestion/extract_text.py`
- Uses **pdfplumber** page by page
- Collects metadata: page number, char count, document author/title
- Output: `pages_text.json` (one entry per page)

#### Step 1b: Text Chunking
- `src/ingestion/chunk_text.py`
- Uses **LangChain's `RecursiveCharacterTextSplitter`**
- Splits long pages into overlapping chunks (default: 800 chars, 100 overlap)
- Overlap means chunks share some text so answers don't get cut off at boundaries
- Each chunk keeps its `page_number` and `content_type` metadata
- Output: `chunks.json`

#### Step 2: Embedding
- `src/embeddings/embed.py`
- Calls **Vertex AI `text-embedding-004`** for each chunk (batched 5 at a time)
- Each chunk becomes a 768-dimensional float vector
- Output: `embeddings.json`

#### Step 2b: FAISS Index
- `src/embeddings/faiss_store.py`
- Builds an in-memory **FAISS** `IndexFlatIP` (exact inner-product search)
- Vectors are L2-normalised so inner-product = cosine similarity
- Saved as `.faiss` (binary index) + `.pkl` (chunk metadata)
- **FAISS is fast but has no built-in filtering by metadata**

#### Step 2c: Qdrant Collection
- `src/embeddings/qdrant_store.py`
- Uploads the same embeddings to a local **Qdrant** instance (Docker)
- Qdrant stores both the vector AND the payload (page number, content type)
- **Qdrant supports metadata filtering** – e.g. "only search pages 10-20"
- FAISS vs Qdrant comparison:
  - FAISS: faster, simpler, no persistence issues, best for prototyping
  - Qdrant: production-ready, supports filters, scales to millions of vectors

#### Step 3: Retrieval
- `src/retrieval/retriever.py`
- Query text is embedded with the same Vertex AI model
- FAISS or Qdrant is searched for top-K similar chunks
- Returns chunks ranked by cosine similarity

#### Step 4: Generation
- `src/generation/generator.py`
- Retrieved chunks are formatted into a prompt with page citations
- **Gemini 2.0 Flash** generates the answer using **streaming** (tokens printed as they arrive)
- Supports **structured JSON output** (answer + sources + confidence)
- **Langfuse tracing**: every query+response is logged automatically

#### Step 4b: Streamlit UI
- `src/ui/streamlit_app.py`
- Clean web interface with sidebar controls
- Shows retrieved context chunks + their scores
- Streams Gemini's answer token by token
- Supports metadata filtering (page range, content type)

**Run command:**
```bash
python scripts/run_phase1.py
# Then:
streamlit run src/ui/streamlit_app.py
```

---

### 📌 Phase 2 – RAG Evaluation

**Goal:** Measure how good the RAG pipeline actually is.

**Evaluation dataset:** `RAG_evaluation_dataset.xlsx`
- 34 questions with ground truth answers and ground truth context
- Columns: Question, Ground_Truth_Context, Ground_Truth_Answer, Page_Number, Context_Content_Type

**Two evaluation methods:**

#### Method 1: RAGAS Framework
- `src/evaluation/ragas_eval.py`
- Metrics:
  - **Faithfulness**: Is the answer grounded in the retrieved context? (no hallucination)
  - **Answer Relevancy**: Does the answer actually address the question?
  - **Context Recall**: Did the retriever find the chunks that contain the answer?
  - **Context Precision**: Are the retrieved chunks relevant (or noisy)?
- Scores are 0.0–1.0, higher is better

#### Method 2: LLM-as-a-Judge
- Gemini reads the question, generated answer, and ground truth
- Scores 1-5 and gives a verdict: correct / partial / incorrect
- Provides a reasoning sentence

**Run command:**
```bash
# Evaluate Phase 1 (FAISS)
python scripts/run_phase2_eval.py --mode faiss

# Evaluate with Qdrant
python scripts/run_phase2_eval.py --mode qdrant

# Evaluate Phase 3 (hybrid + rerank)
python scripts/run_phase2_eval.py --mode hybrid --rerank
```

---

### 📌 Phase 3 – Hybrid Search & Re-ranking

**Goal:** Improve retrieval quality beyond plain vector similarity.

#### Re-ranking (Cross-Encoder)
- `src/retrieval/retriever.py` → `rerank()` function
- First, retrieve top-20 chunks via embedding similarity (fast but approximate)
- Then, run a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) over each (query, chunk) pair
- Cross-encoder reads query + document together → much more accurate relevance score
- Re-rank the 20 candidates and return top-K
- Trade-off: slower, but significantly better precision

#### Metadata Filtering
- Qdrant supports filtering by `page_number` and `content_type`
- Example: "Search only in pages 40-60" (financial statements section)
- Enabled via the Streamlit sidebar's metadata filter controls

#### Sparse Retrieval (BM25)
- `src/retrieval/retriever.py` → `search_bm25()` function
- BM25 is a keyword-based search (like old-school search engines)
- Finds exact keyword matches that embedding search might miss
- Uses `rank-bm25` library

#### Hybrid Retrieval
- Combines dense (embedding) scores + sparse (BM25) scores
- For each chunk: `hybrid_score = (dense_score + bm25_score) / 2`
- Better coverage: catches both semantic matches and exact keyword hits

**Run command:**
```bash
python scripts/run_phase3_hybrid.py
```

---

### 📌 Phase 4 – Advanced RAG Techniques

**Goal:** Speed up the system (semantic cache) and handle complex multi-part questions (multi-hop).

**Run command:**
```bash
python scripts/run_phase4_advanced.py --task both
# Or individually:
python scripts/run_phase4_advanced.py --task cache
python scripts/run_phase4_advanced.py --task multihop
```

**Notebook:** `notebooks/04b_phase4_advanced_rag.ipynb`

---

#### Task 1 – Semantic Caching
- `src/retrieval/semantic_cache.py`

**How it works:**
```
User question
     ↓
Embed question → 768-dim vector
     ↓
Compare with ALL cached vectors (cosine similarity)
     ↓                              ↓
sim ≥ 0.92 → CACHE HIT          sim < 0.92 → CACHE MISS
return instantly                  call Gemini → store result
```

**What is stored per cache entry:**
- `query` – the original question text
- `embedding` – 768-dim float vector
- `answer` – the generated answer
- `timestamp` – when it was stored (for TTL)
- `last_accessed` – when it was last used
- `hits` – how many times this entry was reused
- `source_pages` – which PDF pages contributed to the answer

**Extra features beyond the basics:**
- **TTL (time-to-live)** – entries can expire after N seconds (e.g. `SemanticCache(ttl_seconds=3600)` for 1-hour expiry)
- **Cache warming** – pre-populate on startup with `cache.warm(known_questions, pipeline)` so the first real user request is instant
- **Statistics** – `cache.print_stats()` shows session hit rate, total entries, all-time reuses
- **CSV export** – `cache.export_csv()` dumps the cache to a readable CSV for debugging
- **Similarity introspection** – `cached_rag_query()` returns `(answer, from_cache)` so the UI can show a cache indicator

**Files produced:**
- `data/processed/semantic_cache.json` – the persistent cache
- `data/processed/semantic_cache_export.csv` – human-readable dump

---

#### Task 2 – Multi-hop Retrieval
- `src/retrieval/multihop_retriever.py`

**Why do we need multi-hop?**

A normal (single-hop) RAG does one retrieval step:
```
Question → retrieve 5 chunks → generate answer
```

But some questions span multiple topics and CANNOT be answered from a single retrieval:

> "How did IFC's net income *change* between 2023 and 2024, and what *risk factors* caused this?"

This needs:
- Hop 1: find the 2023 net income figure
- Hop 2: find the 2024 net income figure
- Hop 3: find risk factors relevant to the income change
- Synthesis: combine all three partial answers

**Architecture:**
```
Complex question
       ↓
[Gemini] Decompose → [sub-Q1, sub-Q2, sub-Q3]
       ↓
Hop 1: retrieve(sub-Q1)
       → partial answer 1
       ↓  (partial answer 1 passed as context to next hop)
Hop 2: retrieve(sub-Q2) + context from Hop 1
       → partial answer 2
       ↓
Hop 3: retrieve(sub-Q3) + context from Hops 1 & 2
       → partial answer 3
       ↓
[Gemini] Synthesise all partial answers → FINAL ANSWER
```

**Key design choice – context carry-forward:**
Each hop receives the partial answers from ALL previous hops as extra context. This means Hop 3 already "knows" what Hops 1 and 2 found, so it can ask a more targeted sub-question and avoid redundant retrieval.

**`needs_multihop(question)` detector:**
A heuristic function that scans for keywords indicating multi-hop is needed:
- Comparison words: `change`, `versus`, `between`, `difference`
- Causal words: `why`, `because`, `contributed`, `caused`
- Multi-part words: `both`, `and also`, `as well as`
- Time-span words: `from 2023 to 2024`, `year-over-year`, `trend`

**Classes and functions:**
- `MultiHopRetriever` – orchestrates the full pipeline (decompose → hop loop → synthesise)
- `decompose_question()` – calls Gemini to produce sub-questions as JSON
- `answer_sub_question()` – one retrieval + generation step with prior context
- `synthesise_final_answer()` – final Gemini call to weave all partial answers together
- `MultiHopResult` dataclass – holds the full trace: sub-questions, each hop's chunks and partial answer, final answer
- `Hop` dataclass – one retrieval hop's data

**Trade-offs:**
| Aspect | Single-hop | Multi-hop |
|--------|-----------|-----------|
| Speed | Fast (1 LLM call) | Slower (N+1 LLM calls) |
| Simple questions | ✅ Perfect | ❌ Overkill |
| Multi-part questions | ❌ Misses info | ✅ Better coverage |
| Cost | Low | Higher (multiple calls) |
| Transparency | Low | High (each hop is logged) |

---

### 📌 Phase 5 – Multimodal RAG (Tables + Images)

**Goal:** Make the RAG system understand tables and charts, not just text.

#### Table Retrieval
- `src/ingestion/extract_tables.py` extracts all tables as Markdown strings
- `src/retrieval/multimodal_retriever.py` converts each table into a retrievable chunk
- The chunk text = "Table from page X:\n| col1 | col2 | ... |"
- These table chunks go into the same FAISS/Qdrant index as text chunks
- A query like "What was IFC's net income?" can now retrieve the actual financial table

#### Image Retrieval
- `src/ingestion/extract_images.py` extracts every image from the PDF
- Gemini generates a detailed description of each image: chart type, data shown, trends
- Those captions become chunks (text) in the vector index
- A query like "Describe the portfolio composition chart" can find the right image description

#### Unified Retrieval
- `load_all_chunks_multimodal()` combines text + table + image chunks
- All get embedded and indexed together
- The `content_type` field (`text`, `table`, `image`) enables filtering if needed

**Run command:**
```bash
python scripts/run_phase5_multimodal.py
```

---

### 📌 Phase 6 – ColPali-like Visual RAG

**Goal:** End-to-end visual document understanding – treat the whole page as the unit.

#### Why ColPali?
Classic RAG: PDF → extract text → embed text → search text
ColPali idea: PDF → convert pages to images → embed the visual content → search visually

**Our implementation:**

1. **Page Rasterisation** (`colpali_retriever.py → pdf_pages_to_images`)
   - Convert every PDF page to a PNG at 150 DPI using PyMuPDF

2. **Page Description** (`generate_page_descriptions`)
   - Send each page image to Gemini Vision
   - Gemini describes EVERYTHING: text, tables, charts, numbers
   - This description becomes the retrieval unit (richer than pdfplumber text)

3. **Embedding & Indexing**
   - Embed the page descriptions with the same Vertex AI model
   - Store in a separate FAISS index (so Phase 1's index stays clean)

4. **Visual Answer Generation** (`answer_with_visual_context`)
   - Retrieved top-K page images are passed DIRECTLY to Gemini Vision
   - Gemini sees the actual page layout, charts, and tables
   - Produces answers grounded in visual understanding

5. **Comparison with Phase 1-5:**
   - Phase 1-5 is better for: text-heavy questions, fast retrieval, cost efficiency
   - Phase 6 is better for: chart/graph questions, layout-dependent info, table reading

**Run command:**
```bash
python scripts/run_phase6_colpali.py
# Warning: This makes one Gemini API call per PDF page – can be slow!
```

---

## 🚀 Full Execution Order

```bash
# ── Setup ──────────────────────────────────────────────────────
cp .env.example .env              # Set GCP_PROJECT_ID
cp sweeti.json keys/sweeti.json   # Your GCP key
pip install -r requirements.txt
cd docker && docker-compose up -d && cd ..

# ── Phase 1: Build text RAG ─────────────────────────────────────
python scripts/run_phase1.py
streamlit run src/ui/streamlit_app.py     # http://localhost:8501

# ── Phase 2: Evaluate ───────────────────────────────────────────
python scripts/run_phase2_eval.py --mode faiss

# ── Phase 3: Hybrid + Re-rank ───────────────────────────────────
python scripts/run_phase3_hybrid.py
python scripts/run_phase2_eval.py --mode hybrid --rerank

# ── Phase 5: Multimodal ─────────────────────────────────────────
python scripts/run_phase5_multimodal.py
python scripts/run_phase2_eval.py --mode faiss    # Re-evaluate

# ── Phase 6: ColPali visual RAG ─────────────────────────────────
python scripts/run_phase6_colpali.py

# ── Docker (full containerised app) ─────────────────────────────
docker build -t rag-app .
docker run -p 8501:8501 -v $(pwd)/keys:/app/keys -v $(pwd)/data:/app/data rag-app
```

---

## 🔑 Key Concepts Explained Simply

| Term | What it means |
|------|---------------|
| **Chunk** | A small piece of text (800 chars) from the PDF |
| **Embedding** | A list of 768 numbers that represents the meaning of a chunk |
| **FAISS** | A library that finds the closest number-lists very fast |
| **Qdrant** | A database that stores embeddings + lets you filter by metadata |
| **BM25** | Keyword matching (like Ctrl+F but smarter) |
| **Hybrid** | Combining BM25 keywords + embedding similarity |
| **Re-ranking** | A second, more accurate model checks if chunks are truly relevant |
| **RAG** | Find relevant chunks → paste them in a prompt → let Gemini answer |
| **RAGAS** | A framework that automatically scores how good the RAG answers are |
| **Langfuse** | Records every query and response so you can debug and monitor |
| **ColPali** | Treat whole page images as retrieval units instead of text chunks |
| **Streaming** | Gemini sends words one at a time as it writes them (feels faster) |
| **Semantic Cache** | Save answers; for similar future questions, skip the API call |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `GOOGLE_APPLICATION_CREDENTIALS` error | Make sure `keys/sweeti.json` exists and `.env` has the right path |
| `GCP_PROJECT_ID` error | Set your real project ID in `.env` |
| Qdrant connection refused | Run `docker-compose up -d` in the `docker/` folder |
| FAISS file not found | Run `python scripts/run_phase1.py` first |
| Embedding quota exceeded | Reduce `BATCH_SIZE` in `embed.py` or add `time.sleep()` |
| Streamlit shows empty answer | Check that chunks.json and embeddings.json exist in `data/processed/text/` |
