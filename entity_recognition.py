from gliner2 import AutoExtractor, BoundaryExtractor

NER_MODEL_NAME = "fastino/gliner2.5-base-v1"
NER_THRESHOLD = 0.5
ENTITY_TYPES = {
    "person": (
        "Full names of real human beings only. Exclude identifiers, invoice "
        "numbers, adjacent locations, employers, software, technologies, job "
        "titles, document headings, and organizations."
    ),
    "organization": (
        "Proper names of companies, universities, government bodies, and other "
        "institutions. Exclude software libraries, products, technologies, skills, "
        "journal or publication names, currencies, and headings."
    ),
    "location": (
        "Names of countries, cities, regions, and other geographical places. "
        "Exclude street addresses, organizations, and adjacent person names."
    ),
    "date": "Explicit calendar dates, months, years, durations, or date ranges.",
    "money": (
        "Monetary amounts explicitly containing a currency symbol, currency code, "
        "or currency name. Exclude telephone numbers and standalone numbers."
    ),
}
ENTITY_LABELS = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "GPE",
    "date": "DATE",
    "money": "MONEY",
}


def create_ner_model() -> BoundaryExtractor:
    """Load the local English GLiNER2.5 entity extraction model."""
    return AutoExtractor.from_pretrained(NER_MODEL_NAME)


def extract_entities(text: str, ner_model: BoundaryExtractor) -> list[dict]:
    """Extract described entity types, spans, and confidence from English text."""
    result = ner_model.extract_entities_long(
        text,
        ENTITY_TYPES,
        threshold=NER_THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    return [
        {
            "text": entity["text"],
            "label": ENTITY_LABELS[label],
            "start": entity["start"],
            "end": entity["end"],
            "confidence": entity["confidence"],
        }
        for label, entities in result["entities"].items()
        for entity in entities
    ]
