from io import BytesIO

import numpy as np
import pymupdf
from paddleocr import PaddleOCR
from PIL import Image, UnidentifiedImageError


def create_ocr_engine() -> PaddleOCR:
    """Create a PaddleOCR engine configured for reliable CPU inference."""
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def extract_text(content: bytes, extension: str, ocr_engine: PaddleOCR) -> str:
    """Route document content to the appropriate PDF or image extractor."""
    if extension == ".pdf":
        return extract_pdf_text(content, ocr_engine)
    return extract_image_text(content, ocr_engine)


def extract_pdf_text(content: bytes, ocr_engine: PaddleOCR) -> str:
    """Extract embedded text from a PDF, using OCR for scanned pages."""
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise ValueError("The PDF is invalid or unreadable.") from exc

    pages = []
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
                text = run_ocr(image[:, :, ::-1], ocr_engine)

            if text:
                pages.append(f"Page {page_number}\n{text}")

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")
    return "\n\n".join(pages)


def extract_image_text(content: bytes, ocr_engine: PaddleOCR) -> str:
    """Extract text from an image using PaddleOCR."""
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The image is invalid or unreadable.") from exc

    text = run_ocr(np.asarray(image)[:, :, ::-1], ocr_engine)
    if not text:
        raise ValueError("No text could be extracted from the image.")
    return text


def run_ocr(image: np.ndarray, ocr_engine: PaddleOCR) -> str:
    """Run PaddleOCR on an image and combine recognized lines into plain text."""
    lines = []
    for result in ocr_engine.predict(image):
        data = result.json
        ocr_data = data.get("res", data)
        lines.extend(
            text.strip()
            for text in ocr_data.get("rec_texts", [])
            if text.strip()
        )
    return "\n".join(lines)
