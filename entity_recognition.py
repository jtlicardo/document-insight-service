import en_core_web_sm
from spacy.language import Language


def create_ner_model() -> Language:
    """Load spaCy's small English pipeline with only NER-related components."""
    return en_core_web_sm.load(
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
    )


def extract_entities(text: str, ner_model: Language) -> list[dict]:
    """Extract named entities and their character positions from English text."""
    document = ner_model(text)
    return [
        {
            "text": entity.text,
            "label": entity.label_,
            "start": entity.start_char,
            "end": entity.end_char,
        }
        for entity in document.ents
    ]
