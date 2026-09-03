import os

from openai import OpenAI
from pydantic import BaseModel


class QuestionAnswer(BaseModel):
    """A grounded answer and the retrieved sources that support it."""

    answer: str
    source_ids: list[str]


def answer_question(question: str, chunks: list[dict]) -> QuestionAnswer:
    """Answer a question and identify the chunks that support the answer."""
    document_text = "\n\n".join(
        f"Source ID: {chunk['source_id']}\n"
        f"Document: {chunk['filename']}\n"
        f"Excerpt:\n{chunk['text']}"
        for chunk in chunks
    )

    client = OpenAI()
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        instructions=(
            "Answer using only the supplied document excerpts. "
            "Return only the source IDs of excerpts that directly support the answer. "
            "If the answer is not in the excerpts, say that it could not be found and "
            "return an empty source_ids list."
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
