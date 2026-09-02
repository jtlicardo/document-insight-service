from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Document Insight Service")
app.state.uploaded_documents = []

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    documents = []
    for file in files:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension or 'unknown'}",
            )

        documents.append(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": await file.read(),
            }
        )

    request.app.state.uploaded_documents = documents

    return {
        "uploaded_files": [document["filename"] for document in documents],
        "message": "Files uploaded successfully.",
    }


@app.post("/ask")
def ask_question(question_request: QuestionRequest, request: Request):
    if not request.app.state.uploaded_documents:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    return {
        "question": question_request.question,
        "answer": "Question answering is not implemented yet.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
