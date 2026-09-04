import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from openai import OpenAIError

from config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_COUNT,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
)
from document_processing import create_ocr_engine, extract_text
from entity_recognition import create_ner_model, extract_entities
from question_answering import answer_question
from retrieval import create_document_chunks, retrieve_relevant_chunks
from schemas import AskResponse, IndexResponse, QuestionRequest, UploadResponse

app = FastAPI(title="Document insight service")
app.state.uploaded_documents = []
app.state.document_chunks = []
app.state.ocr_engine = None
app.state.ner_model = None
logger = logging.getLogger("uvicorn.error")


def elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds rounded for concise logging."""
    return round((perf_counter() - started_at) * 1_000, 2)


def get_ner_model(request: Request):
    """Load and reuse the English NER model."""
    if request.app.state.ner_model is None:
        request.app.state.ner_model = create_ner_model()
    return request.app.state.ner_model


def create_document_index(request: Request) -> int:
    """Create the current document index."""
    request.app.state.document_chunks = create_document_chunks(
        request.app.state.uploaded_documents
    )
    return len(request.app.state.document_chunks)


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File()],
) -> UploadResponse:
    """Extract and temporarily store text from uploaded PDF or image documents."""
    upload_started = perf_counter()
    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_FILE_COUNT} documents can be uploaded at once.",
        )

    def get_ocr_engine():
        """Create the OCR engine only when a document actually needs it."""
        if request.app.state.ocr_engine is None:
            request.app.state.ocr_engine = create_ocr_engine()
        return request.app.state.ocr_engine

    documents = []
    for file in files:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension or 'unknown'}",
            )

        content = await file.read(MAX_FILE_SIZE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail=f"{file.filename} is empty.")
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{file.filename} exceeds the {MAX_FILE_SIZE_MB} MB "
                    "per-document limit."
                ),
            )

        try:
            extracted_document = extract_text(content, extension, get_ocr_engine)
        except ValueError as exc:
            logger.warning("Upload failed: invalid document")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        entities = extract_entities(
            extracted_document["text"],
            get_ner_model(request),
        )
        character_count = len(extracted_document["text"])

        documents.append(
            {
                "filename": file.filename,
                "text": extracted_document["text"],
                "pages": extracted_document["pages"],
                "entities": entities,
                "page_count": extracted_document["page_count"],
                "ocr_pages": extracted_document["ocr_pages"],
                "character_count": character_count,
            }
        )

    request.app.state.uploaded_documents = documents
    request.app.state.document_chunks = []
    logger.info(
        "Upload complete: documents=%d pages=%d ocr_pages=%d duration_ms=%.2f",
        len(documents),
        sum(document["page_count"] for document in documents),
        sum(document["ocr_pages"] for document in documents),
        elapsed_ms(upload_started),
    )

    return UploadResponse(
        documents=[
            {
                "filename": document["filename"],
                "text": document["text"],
                "entities": document["entities"],
                "page_count": document["page_count"],
                "ocr_pages": document["ocr_pages"],
                "character_count": document["character_count"],
            }
            for document in documents
        ],
        message="Files uploaded successfully.",
    )


@app.post("/index", response_model=IndexResponse)
def index_documents(request: Request) -> IndexResponse:
    """Create and temporarily store embeddings for the uploaded documents."""
    indexing_started = perf_counter()
    if not request.app.state.uploaded_documents:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured.",
        )

    try:
        chunk_count = create_document_index(request)
    except ValueError as exc:
        logger.warning("Index failed: no extractable text")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIError as exc:
        logger.warning("Index failed: OpenAI request error")
        raise HTTPException(
            status_code=502,
            detail="OpenAI could not create document embeddings.",
        ) from exc

    logger.info(
        "Index complete: documents=%d chunks=%d duration_ms=%.2f",
        len(request.app.state.uploaded_documents),
        chunk_count,
        elapsed_ms(indexing_started),
    )

    return IndexResponse(
        message="Document embeddings created successfully.",
        chunk_count=chunk_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(question_request: QuestionRequest, request: Request) -> AskResponse:
    """Accept a question about the currently uploaded documents."""
    question_started = perf_counter()
    if not request.app.state.uploaded_documents:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured.",
        )

    try:
        if not request.app.state.document_chunks:
            create_document_index(request)

        relevant_chunks = retrieve_relevant_chunks(
            question_request.question,
            request.app.state.document_chunks,
        )
        answer_result = answer_question(
            question_request.question,
            relevant_chunks,
        )
    except (OpenAIError, ValueError) as exc:
        logger.warning("Question failed: OpenAI request error")
        raise HTTPException(
            status_code=502,
            detail="OpenAI could not answer the question.",
        ) from exc

    chunks_by_source_id = {chunk["source_id"]: chunk for chunk in relevant_chunks}
    source_ids = list(dict.fromkeys(answer_result.source_ids))
    sources = [
        {
            "filename": chunks_by_source_id[source_id]["filename"],
            "page_number": chunks_by_source_id[source_id]["page_number"],
            "excerpt": chunks_by_source_id[source_id]["text"],
        }
        for source_id in source_ids
        if source_id in chunks_by_source_id
    ]
    logger.info(
        "Question answered: retrieved_chunks=%d cited_sources=%d duration_ms=%.2f",
        len(relevant_chunks),
        len(sources),
        elapsed_ms(question_started),
    )

    return AskResponse(
        question=question_request.question,
        answer=answer_result.answer,
        sources=sources,
        metadata={
            "retrieved_chunks": len(relevant_chunks),
            "cited_sources": len(sources),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
