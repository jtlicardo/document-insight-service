from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Document Insight Service")


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    return {
        "uploaded_files": [file.filename for file in files],
        "message": "Files received. Document processing is not implemented yet.",
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    return {
        "question": request.question,
        "answer": "Question answering is not implemented yet.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
