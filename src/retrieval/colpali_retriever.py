"""
src/retrieval/colpali_retriever.py
────────────────────────────────────
Phase 6 – ColPali-like Visual RAG

Steps:
  1. Rasterise every PDF page to PNG  (pdf_pages_to_images)
  2. Ask Gemini Vision to describe each page  (generate_page_descriptions)
  3. Embed the descriptions and build a FAISS index  (in run_phase6)
  4. At query time: retrieve top-K page images then pass them directly
     to Gemini Vision for visual-grounded answers  (answer_with_visual_context)

Note: `types.Part.from_text()` was removed in google-genai>=0.5.
      Use plain strings in the contents list instead.
"""

import json
from pathlib import Path

import fitz          # PyMuPDF
from tqdm import tqdm
from google import genai
from google.genai import types

from config.settings import (
    PDF_PATH, DATA_IMAGES,
    GEMINI_MODEL, GCP_PROJECT_ID, GCP_LOCATION,
)
from config.gcp_auth import init_vertex


# ── Step 1: Rasterise PDF pages ───────────────────────────────────────────

def pdf_pages_to_images(
    pdf_path : Path = PDF_PATH,
    out_dir  : Path = None,
    dpi      : int  = 150,
) -> list[dict]:
    """
    Convert every PDF page to a PNG image.

    Returns a list of dicts:
      page_number, image_path, width, height
    """
    if out_dir is None:
        out_dir = DATA_IMAGES / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc   = fitz.open(str(pdf_path))
    pages = []

    for page_num in tqdm(range(len(doc)), desc="Rasterising pages"):
        page = doc[page_num]
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat)

        img_path = out_dir / f"page_{page_num + 1:03d}.png"
        pix.save(str(img_path))

        pages.append({
            "page_number": page_num + 1,
            "image_path" : str(img_path),
            "width"      : pix.width,
            "height"     : pix.height,
        })

    doc.close()
    print(f"✅ Rasterised {len(pages)} pages → {out_dir}")
    return pages


# ── Step 2: Gemini Vision page descriptions ───────────────────────────────

def _describe_page(image_path: str, page_num: int, client) -> str:
    """Ask Gemini Vision to produce a comprehensive description of one page."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    prompt = (
        f"You are analysing page {page_num} of the IFC 2024 Annual Report.\n"
        "Describe EVERYTHING visible on this page:\n"
        "- All headings, paragraphs, and text content\n"
        "- All tables (include the numbers you can read)\n"
        "- All charts and graphs (describe what data they show)\n"
        "- Any highlighted or bold text\n\n"
        "Write a thorough description so someone can answer questions "
        "about this page without seeing the original image."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,             # plain string — no Part.from_text needed
            ],
        )
        return response.text.strip()
    except Exception as e:
        return f"[Description error page {page_num}: {e}]"


def generate_page_descriptions(
    pages   : list[dict],
    out_dir : Path = DATA_IMAGES,
) -> list[dict]:
    """
    Generate Gemini descriptions for all rasterised pages.
    Saves results to data/processed/images/page_descriptions.json.
    """
    init_vertex()
    client = genai.Client(
        vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION
    )

    descriptions = []
    for page in tqdm(pages, desc="Describing pages with Gemini"):
        desc = _describe_page(page["image_path"], page["page_number"], client)

        descriptions.append({
            **page,
            "description" : desc,
            "content_type": "page_image",
            "source_file" : PDF_PATH.name,
            "text"        : f"Page {page['page_number']} visual description:\n{desc}",
            "chunk_id"    : f"page_{page['page_number']:03d}",
        })
        print(f"   p{page['page_number']:03d}: {desc[:80]}...")

    out_file = out_dir / "page_descriptions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(descriptions)} page descriptions → {out_file}")
    return descriptions


def load_page_descriptions(out_dir: Path = DATA_IMAGES) -> list[dict]:
    """Load previously generated page descriptions from disk."""
    desc_file = out_dir / "page_descriptions.json"
    if not desc_file.exists():
        raise FileNotFoundError(
            f"{desc_file} not found.\n"
            "Run `python scripts/run_phase6_colpali.py` first."
        )
    with open(desc_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Step 4: Visual answer generation ─────────────────────────────────────

def answer_with_visual_context(
    query          : str,
    retrieved_pages: list[dict],
    max_pages      : int = 3,
) -> str:
    """
    Pass the retrieved page images DIRECTLY to Gemini Vision
    and ask it to answer the query from what it sees.

    This is the 'late interaction' step of ColPali-style RAG –
    the model grounds its answer in the actual visual content
    of the pages, not just text extracted from them.

    Parameters
    ----------
    query           : the user's question
    retrieved_pages : list of page dicts (must have 'image_path')
    max_pages       : cap how many images are sent (API token limit)
    """
    init_vertex()
    client = genai.Client(
        vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION
    )

    # Build content list: alternating (image_bytes, caption_string) pairs
    contents = []
    pages_used = []

    for page in retrieved_pages[:max_pages]:
        img_path = page.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        )
        # Plain string caption — Part.from_text() is deprecated
        contents.append(
            f"[Above: Page {page['page_number']} of the IFC 2024 Annual Report]"
        )
        pages_used.append(page["page_number"])

    if not contents:
        return "No page images found. Run pdf_pages_to_images() first."

    # Final instruction string
    contents.append(
        f"\nQuestion: {query}\n\n"
        "Answer based ONLY on the page images shown above. "
        f"Cite the specific page numbers (pages shown: {pages_used})."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


if __name__ == "__main__":
    pages        = pdf_pages_to_images(dpi=150)
    descriptions = generate_page_descriptions(pages)
    print(f"Total pages described: {len(descriptions)}")
