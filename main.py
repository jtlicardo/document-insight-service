import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from openai import OpenAIError
from pydantic import BaseModel

from document_processing import create_ocr_engine, extract_text
from entity_recognition import create_ner_model, extract_entities
from question_answering import answer_question
from retrieval import create_document_chunks, retrieve_relevant_chunks

load_dotenv()

app = FastAPI(title="Document insight service")
app.state.uploaded_documents = []
app.state.document_chunks = []
app.state.ocr_engine = None
app.state.ner_model = None

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class QuestionRequest(BaseModel):
    question: str


def get_ner_model(request: Request):
    """Load and reuse the English NER model."""
    if request.app.state.ner_model is None:
        request.app.state.ner_model = create_ner_model()
    return request.app.state.ner_model


@app.post("/upload")
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File()],
):
    """Extract and temporarily store text from uploaded PDF or image documents."""

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

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{file.filename} is empty.")

        try:
            text = extract_text(content, extension, get_ocr_engine)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        documents.append(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
                "text": text,
                "entities": extract_entities(text, get_ner_model(request)),
            }
        )

    request.app.state.uploaded_documents = documents
    request.app.state.document_chunks = []

    return {
        "documents": [
            {
                "filename": document["filename"],
                "text": document["text"],
                "entities": document["entities"],
            }
            for document in documents
        ],
        "message": "Files uploaded successfully.",
    }


@app.post("/index")
def index_documents(request: Request):
    """Create and temporarily store embeddings for the uploaded documents."""
    if not request.app.state.uploaded_documents:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured.",
        )

    try:
        request.app.state.document_chunks = create_document_chunks(
            request.app.state.uploaded_documents
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenAI could not create document embeddings.",
        ) from exc

    return {"message": "Document embeddings created successfully."}


@app.post("/ask")
def ask_question(question_request: QuestionRequest, request: Request):
    """Accept a question about the currently uploaded documents."""
    if not request.app.state.uploaded_documents:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured.",
        )

    try:
        if not request.app.state.document_chunks:
            request.app.state.document_chunks = create_document_chunks(
                request.app.state.uploaded_documents
            )

        relevant_chunks = retrieve_relevant_chunks(
            question_request.question,
            request.app.state.document_chunks,
        )
        answer_result = answer_question(
            question_request.question,
            relevant_chunks,
        )
    except (OpenAIError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenAI could not answer the question.",
        ) from exc

    chunks_by_source_id = {chunk["source_id"]: chunk for chunk in relevant_chunks}
    source_ids = list(dict.fromkeys(answer_result.source_ids))
    sources = [
        {
            "filename": chunks_by_source_id[source_id]["filename"],
            "excerpt": chunks_by_source_id[source_id]["text"],
        }
        for source_id in source_ids
        if source_id in chunks_by_source_id
    ]

    return {
        "question": question_request.question,
        "answer": answer_result.answer,
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
