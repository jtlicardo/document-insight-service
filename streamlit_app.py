import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


ENTITY_COLORS = {
    "PERSON": "blue",
    "ORG": "violet",
    "GPE": "green",
    "LOC": "green",
    "DATE": "orange",
    "TIME": "orange",
    "MONEY": "yellow",
    "PERCENT": "red",
}
ENTITY_LABELS = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Countries and cities",
    "LOC": "Locations",
    "DATE": "Dates",
    "TIME": "Times",
    "MONEY": "Money",
    "PERCENT": "Percentages",
}


def escape_markdown(text: str) -> str:
    """Escape characters that could break Streamlit's inline Markdown syntax."""
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def highlight_entities(text: str, entities: list[dict]) -> str:
    """Add colored Streamlit Markdown highlights to named entities in text."""
    highlighted_text = []
    cursor = 0

    for entity in entities:
        highlighted_text.append(text[cursor : entity["start"]])
        color = ENTITY_COLORS.get(entity["label"], "gray")
        entity_text = escape_markdown(text[entity["start"] : entity["end"]])
        highlighted_text.append(
            f":{color}-background[{entity_text}] "
            f":gray-badge[{entity['label']}]"
        )
        cursor = entity["end"]

    highlighted_text.append(text[cursor:])
    return "".join(highlighted_text)


def display_entity_summary(entities: list[dict]) -> None:
    """Display useful document entities as grouped, color-coded chips."""
    grouped_entities = {}
    for entity in entities:
        if entity["label"] not in ENTITY_LABELS:
            continue
        grouped_entities.setdefault(entity["label"], [])
        if entity["text"] not in grouped_entities[entity["label"]]:
            grouped_entities[entity["label"]].append(entity["text"])

    for label, values in grouped_entities.items():
        display_label = ENTITY_LABELS[label]
        color = ENTITY_COLORS[label]
        entity_chips = " ".join(
            f":{color}-badge[{escape_markdown(value)}]" for value in values
        )
        st.caption(display_label)
        st.markdown(entity_chips)


st.set_page_config(
    page_title="Document insight service",
    page_icon=":material/document_search:",
    layout="centered",
)
st.title("Document insight service")
st.markdown("Turn PDFs and images into answers you can trust.")
st.caption(
    ":material/model_training: **Model stack:** GPT-5.6 Luna · "
    "text-embedding-3-small · spaCy en_core_web_sm"
)
st.caption(
    ":material/language: Works best with English-language documents. Results may "
    "be less accurate in other languages."
)
st.session_state.setdefault("extracted_documents", [])
st.session_state.setdefault("messages", [])

with st.container(border=True, gap="small"):
    st.markdown("### Add documents")
    st.caption("PDF, PNG, JPG or TIFF · multiple files supported")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    upload_clicked = st.button(
        "Process documents",
        type="primary",
        icon=":material/arrow_upward:",
    )

if upload_clicked:
    if not uploaded_files:
        st.warning("Please select at least one document.")
    else:
        document_count = len(uploaded_files)
        files = [
            ("files", (file.name, file.getvalue(), file.type))
            for file in uploaded_files
        ]
        st.session_state["documents_uploaded"] = False
        st.session_state["extracted_documents"] = []
        st.session_state["messages"] = []

        extraction_status = st.status(
            "Extracting text and detecting entities...",
            type="step",
        )
        current_status = extraction_status
        try:
            response = requests.post(f"{API_URL}/upload", files=files, timeout=300)
            response.raise_for_status()
            result = response.json()
            extraction_status.update(
                label="Text and entities extracted",
                state="complete",
            )

            embedding_status = st.status("Creating embeddings...", type="step")
            current_status = embedding_status
            response = requests.post(f"{API_URL}/index", timeout=60)
            response.raise_for_status()
            embedding_status.update(label="Embeddings created", state="complete")

            st.session_state["documents_uploaded"] = True
            st.session_state["extracted_documents"] = result["documents"]
            st.status(
                f"{document_count}/{document_count} document(s) ready",
                state="complete",
                type="step",
            )
            st.success(result["message"])
        except requests.RequestException as exc:
            current_status.update(label="Document processing failed", state="error")
            st.error(f"Could not upload documents: {exc}")

if st.session_state["extracted_documents"]:
    st.subheader("Document insights", anchor=False)
    for index, document in enumerate(st.session_state["extracted_documents"]):
        with st.container(border=True, key=f"document_{index}", gap="small"):
            visible_entities = [
                entity
                for entity in document["entities"]
                if entity["label"] in ENTITY_LABELS
            ]
            entity_count = len({entity["text"] for entity in visible_entities})
            st.markdown(
                f"### :material/description: {escape_markdown(document['filename'])} "
                f":gray-badge[{entity_count} entities]"
            )
            if visible_entities:
                display_entity_summary(visible_entities)
            else:
                st.caption("No useful named entities detected.")

            with st.expander(":material/article: View extracted text"):
                st.text(document["text"])

st.subheader("Ask your documents", anchor=False)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message.get("entities"):
            st.markdown(highlight_entities(message["content"], message["entities"]))
        else:
            st.write(message["content"])

question = st.chat_input(
    "Ask a question about your documents",
    disabled=not st.session_state.get("documents_uploaded", False),
    submit_mode="disable",
)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with (
        st.chat_message("assistant"),
        st.spinner("Finding the answer...", show_time=True),
    ):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "entities": result["entities"],
                }
            )
            if result["entities"]:
                st.markdown(
                    highlight_entities(result["answer"], result["entities"])
                )
            else:
                st.write(result["answer"])
        except requests.RequestException as exc:
            st.error(f"Could not get an answer: {exc}")
