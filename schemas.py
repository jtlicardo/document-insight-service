from pydantic import BaseModel, Field, field_validator

from config import MAX_QUESTION_LENGTH


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        """Strip surrounding whitespace before validating question length."""
        return value.strip() if isinstance(value, str) else value


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class DocumentResponse(BaseModel):
    filename: str
    text: str
    entities: list[EntityResponse]


class UploadResponse(BaseModel):
    documents: list[DocumentResponse]
    message: str


class MessageResponse(BaseModel):
    message: str


class SourceResponse(BaseModel):
    filename: str
    page_number: int | None
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
