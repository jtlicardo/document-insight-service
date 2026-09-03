from collections.abc import Callable
from io import BytesIO
from typing import TypedDict

import numpy as np
import pymupdf
from paddleocr import PaddleOCR
from PIL import Image, UnidentifiedImageError


class ExtractedPage(TypedDict):
    page_number: int | None
    text: str


class ExtractedDocument(TypedDict):
    text: str
    pages: list[ExtractedPage]


def create_ocr_engine() -> PaddleOCR:
    """Create a PaddleOCR engine configured for reliable CPU inference."""
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def extract_text(
    content: bytes,
    extension: str,
    get_ocr_engine: Callable[[], PaddleOCR],
) -> ExtractedDocument:
    """Extract document text while preserving available page metadata."""
    if extension == ".pdf":
        return extract_pdf_text(content, get_ocr_engine)
    return extract_image_text(content, get_ocr_engine())


def extract_pdf_text(
    content: bytes,
    get_ocr_engine: Callable[[], PaddleOCR],
) -> ExtractedDocument:
    """Extract PDF text by page, using OCR when embedded text is unavailable."""
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise ValueError("The PDF is invalid or unreadable.") from exc

    pages: list[ExtractedPage] = []
    with document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height,
                    pixmap.width,
                    pixmap.n,
                )
                text = run_ocr(image[:, :, ::-1], get_ocr_engine())

            if text:
                pages.append({"page_number": page_number, "text": text})

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")
    return {
        "text": "\n\n".join(
            f"Page {page['page_number']}\n{page['text']}" for page in pages
        ),
        "pages": pages,
    }


def extract_image_text(content: bytes, ocr_engine: PaddleOCR) -> ExtractedDocument:
    """Extract image text with no PDF page number using PaddleOCR."""
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The image is invalid or unreadable.") from exc

    text = run_ocr(np.asarray(image)[:, :, ::-1], ocr_engine)
    if not text:
        raise ValueError("No text could be extracted from the image.")
    return {
        "text": text,
        "pages": [{"page_number": None, "text": text}],
    }


def run_ocr(image: np.ndarray, ocr_engine: PaddleOCR) -> str:
    """Run PaddleOCR on an image and combine recognized lines into plain text."""
    lines = []
    for result in ocr_engine.predict(image):
        data = result.json
        ocr_data = data.get("res", data)
        lines.extend(
            text.strip() for text in ocr_data.get("rec_texts", []) if text.strip()
        )
    return "\n".join(lines)
