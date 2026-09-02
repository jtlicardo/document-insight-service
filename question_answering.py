import os

from openai import OpenAI


def answer_question(question: str, chunks: list[dict]) -> str:
    """Answer a question using the most relevant document chunks."""
    document_text = "\n\n".join(
        f"Document: {chunk['filename']}\n{chunk['text']}" for chunk in chunks
    )

    client = OpenAI()
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        instructions=(
            "Answer using only the supplied document text. "
            "If the answer is not in the documents, say that it could not be found."
        ),
        input=f"Document text:\n{document_text}\n\nQuestion: {question}",
        reasoning={"effort": "low"},
        max_output_tokens=500,
        store=False,
    )

    return response.output_text
