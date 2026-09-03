import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Document insight service")
st.session_state.setdefault("extracted_documents", [])

uploaded_files = st.file_uploader(
    "Upload PDF or image documents",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
)

if st.button("Upload documents"):
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

        extraction_status = st.status("Extracting text...", type="step")
        current_status = extraction_status
        try:
            response = requests.post(f"{API_URL}/upload", files=files, timeout=300)
            response.raise_for_status()
            result = response.json()
            extraction_status.update(label="Text extracted", state="complete")

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
    st.subheader("Extracted text")
    for document in st.session_state["extracted_documents"]:
        with st.expander(document["filename"]):
            st.text(document["text"])

question = st.text_input("Ask a question about the documents")

if st.button("Ask"):
    if not st.session_state.get("documents_uploaded"):
        st.warning("Please upload at least one document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your question...", show_time=True):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                st.subheader("Answer")
                st.write(result["answer"])
            except requests.RequestException as exc:
                st.error(f"Could not get an answer: {exc}")
