import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Document Insight Service")

uploaded_files = st.file_uploader(
    "Upload PDF or image documents",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if st.button("Upload documents"):
    if not uploaded_files:
        st.warning("Please select at least one document.")
    else:
        files = [
            ("files", (file.name, file.getvalue(), file.type))
            for file in uploaded_files
        ]

        try:
            response = requests.post(f"{API_URL}/upload", files=files, timeout=30)
            response.raise_for_status()
            result = response.json()

            st.session_state["documents_uploaded"] = True
            st.success(result["message"])
            st.write("Uploaded files:", result["uploaded_files"])
        except requests.RequestException as exc:
            st.error(f"Could not upload documents: {exc}")

question = st.text_input("Ask a question about the documents")

if st.button("Ask"):
    if not st.session_state.get("documents_uploaded"):
        st.warning("Please upload at least one document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
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
