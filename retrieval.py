import os

import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_CHUNKS = 5


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping token-based chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    step = CHUNK_SIZE - CHUNK_OVERLAP

    return [
        encoding.decode(tokens[start : start + CHUNK_SIZE])
        for start in range(0, len(tokens), step)
    ]


def create_document_chunks(documents: list[dict]) -> list[dict]:
    """Chunk document text and create an embedding for every chunk."""
    chunks = [
        {"filename": document["filename"], "text": chunk}
        for document in documents
        for chunk in chunk_text(document["text"])
    ]

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded documents.")

    client = OpenAI()
    response = client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=[chunk["text"] for chunk in chunks],
    )

    for item in response.data:
        chunks[item.index]["embedding"] = item.embedding

    return chunks


def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
) -> list[dict]:
    """Return the document chunks most semantically similar to a question."""
    client = OpenAI()
    response = client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
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
            "similarity": float(similarities[index]),
        }
        for source_number, index in enumerate(top_indices, start=1)
    ]
