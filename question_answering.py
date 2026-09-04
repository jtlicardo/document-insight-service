from openai import OpenAI
from pydantic import BaseModel, Field

from config import OPENAI_MODEL
from retrieval import RetrievedChunk


class QuestionAnswer(BaseModel):
    """A grounded answer and the retrieved sources that support it."""

    answer: str = Field(
        description=(
            "A direct, concise natural-language answer to the user's question. "
            "Never return source IDs in this field."
        )
    )
    source_ids: list[str] = Field(
        description=(
            "The source IDs of excerpts that directly support the answer. "
            "Return an empty list when the answer is not present."
        )
    )


def answer_question(question: str, chunks: list[RetrievedChunk]) -> QuestionAnswer:
    """Answer a question and identify the chunks that support the answer."""
    document_text = "\n\n".join(
        f"Source ID: {chunk['source_id']}\n"
        f"Document: {chunk['filename']}\n"
        f"Page: {chunk['page_number'] or 'Not applicable'}\n"
        f"Excerpt:\n{chunk['text']}"
        for chunk in chunks
    )

    client = OpenAI()
    response = client.responses.parse(
        model=OPENAI_MODEL,
        instructions=(
            "Answer the user's question directly in natural language, using only the "
            "supplied document excerpts. Select only excerpts that directly support "
            "the answer. If the answer is not present, say that it could not be found."
        ),
        input=f"Document excerpts:\n{document_text}\n\nQuestion: {question}",
        reasoning={"effort": "low"},
        max_output_tokens=500,
        store=False,
        text_format=QuestionAnswer,
    )

    if response.output_parsed is None:
        raise ValueError("OpenAI returned an answer that could not be parsed.")
    return response.output_parsed
