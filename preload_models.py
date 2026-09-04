from document_processing import create_ocr_engine
from entity_recognition import create_ner_model


def main() -> None:
    """Download the local NER and OCR model files into their caches."""
    print("Preloading GLiNER model...")
    create_ner_model()

    print("Preloading PaddleOCR models...")
    create_ocr_engine()

    print("Model files are ready.")


if __name__ == "__main__":
    main()
