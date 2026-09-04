from typing import TypedDict

import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_EMBEDDING_MODEL

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_CHUNKS = 5


class RetrievalPage(TypedDict):
    page_number: int | None
    text: str


class RetrievalDocument(TypedDict):
    filename: str
    pages: list[RetrievalPage]


class DocumentChunk(TypedDict):
    filename: str
    page_number: int | None
    text: str
    embedding: list[float]


class RetrievedChunk(DocumentChunk):
    source_id: str


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping token-based chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    step = CHUNK_SIZE - CHUNK_OVERLAP

    return [
        encoding.decode(tokens[start : start + CHUNK_SIZE])
        for start in range(0, len(tokens), step)
    ]


def create_document_chunks(
    documents: list[RetrievalDocument],
) -> list[DocumentChunk]:
    """Chunk document text and create an embedding for every chunk."""
    chunk_texts = [
        {
            "filename": document["filename"],
            "page_number": page["page_number"],
            "text": chunk,
        }
        for document in documents
        for page in document["pages"]
        for chunk in chunk_text(page["text"])
    ]

    if not chunk_texts:
        raise ValueError("No text could be extracted from the uploaded documents.")

    client = OpenAI()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=[chunk["text"] for chunk in chunk_texts],
    )
    embeddings = {item.index: item.embedding for item in response.data}

    return [
        {**chunk, "embedding": embeddings[index]}
        for index, chunk in enumerate(chunk_texts)
    ]


def retrieve_relevant_chunks(
    question: str,
    chunks: list[DocumentChunk],
) -> list[RetrievedChunk]:
    """Return the document chunks most semantically similar to a question."""
    client = OpenAI()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=question,
    )
    question_embedding = response.data[0].embedding
    chunk_embeddings = [chunk["embedding"] for chunk in chunks]
    similarities = cosine_similarity(
        [question_embedding],
        chunk_embeddings,
    )[0]
    top_indices = similarities.argsort()[::-1][:TOP_CHUNKS]

    return [
        {
            **chunks[index],
            "source_id": f"S{source_number}",
        }
        for source_number, index in enumerate(top_indices, start=1)
    ]
