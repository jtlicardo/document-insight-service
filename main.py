from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from document_processing import create_ocr_engine, extract_text

app = FastAPI(title="Document Insight Service")
app.state.uploaded_documents = []
app.state.ocr_engine = None

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File()],
):
    """Extract and temporarily store text from uploaded PDF or image documents."""
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

        if request.app.state.ocr_engine is None:
            request.app.state.ocr_engine = create_ocr_engine()

        try:
            text = extract_text(content, extension, request.app.state.ocr_engine)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        documents.append(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
                "text": text,
            }
        )

    request.app.state.uploaded_documents = documents

    return {
        "documents": [
            {"filename": document["filename"], "text": document["text"]}
            for document in documents
        ],
        "message": "Files uploaded successfully.",
    }


@app.post("/ask")
def ask_question(question_request: QuestionRequest, request: Request):
    """Accept a question about the currently uploaded documents."""
    if not request.app.state.uploaded_documents:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    return {
        "question": question_request.question,
        "answer": "Question answering is not implemented yet.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
