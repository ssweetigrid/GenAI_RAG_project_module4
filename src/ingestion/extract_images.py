"""
src/ingestion/extract_images.py
────────────────────────────────
Practice 1 – Data Parsing Requirement 2
Extract ALL images from the IFC PDF, then ask Gemini to write
a descriptive caption for each one.

Libraries:
  • PyMuPDF (fitz) – fast image extraction with position info
  • google-genai   – Gemini 2.0 Flash for captioning
"""

import json
import base64
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm
from google import genai
from google.genai import types

from config.settings import (
    PDF_PATH, DATA_IMAGES,
    GEMINI_MODEL, GCP_PROJECT_ID, GCP_LOCATION,
)
from config.gcp_auth import init_vertex


# Minimum pixel area to keep an image (skip tiny icons/decorations)
MIN_AREA = 5000


def _get_gemini_client():
    """Create a Vertex-AI–backed Gemini client."""
    init_vertex()
    return genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)


def _gemini_caption(client, image_bytes: bytes, page_num: int, img_idx: int) -> str:
    """
    Send the image to Gemini and ask for a detailed caption.
    Returns the caption string.
    """
    # Encode image as base64 for the API
    b64 = base64.standard_b64encode(image_bytes).decode()

    prompt = (
        "You are analysing a figure extracted from the IFC 2024 Annual Report. "
        "Describe this image in detail: what type of chart/graph/figure is it, "
        "what data does it show, what are the key numbers or trends visible, "
        "and what financial insight does it convey? "
        "Write a comprehensive paragraph (5-8 sentences)."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        return response.text.strip()
    except Exception as e:
        return f"[Caption error: {e}]"


def extract_images_from_pdf(
    pdf_path: Path = PDF_PATH,
    caption: bool = True,
) -> list[dict]:
    """
    Returns a list of image dicts:
      - page_number   : int
      - image_index   : int
      - width, height : int
      - file_path     : str  (saved PNG path)
      - caption       : str  (Gemini-generated description)
      - content_type  : "image"
      - source_file   : str
    """
    out_dir = DATA_IMAGES
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _get_gemini_client() if caption else None

    doc = fitz.open(str(pdf_path))
    all_images = []

    for page_num in tqdm(range(len(doc)), desc="Extracting images"):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            # Extract raw image bytes
            base_image = doc.extract_image(xref)
            img_bytes   = base_image["image"]
            img_ext     = base_image["ext"]          # png / jpeg / etc.
            width       = base_image["width"]
            height      = base_image["height"]

            # Skip tiny images (icons, bullets, borders)
            if width * height < MIN_AREA:
                continue

            # Save image to disk
            img_filename = f"page{page_num+1:03d}_img{img_idx:02d}.{img_ext}"
            img_path = out_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # Generate caption
            cap = ""
            if caption and client:
                print(f"  🤖 Captioning {img_filename}...")
                cap = _gemini_caption(client, img_bytes, page_num + 1, img_idx)

            all_images.append({
                "page_number" : page_num + 1,
                "image_index" : img_idx,
                "width"       : width,
                "height"      : height,
                "file_path"   : str(img_path),
                "caption"     : cap,
                "source_file" : pdf_path.name,
                "content_type": "image",
            })

    doc.close()
    print(f"✅ Extracted {len(all_images)} images")
    return all_images


def save_extracted_images(images: list[dict], out_dir: Path = DATA_IMAGES):
    """Save image metadata (with captions) to a JSON file."""
    json_file = out_dir / "images_metadata.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved image metadata → {json_file}")
    return json_file


if __name__ == "__main__":
    images = extract_images_from_pdf(caption=True)
    save_extracted_images(images)
