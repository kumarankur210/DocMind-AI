
import os
import tempfile

import streamlit as st

from rag_pipeline import process_pdf, ask_question


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


# ---------------- SESSION STATE ----------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# ---------------- HEADER ----------------

st.title("📚 DocuMind AI")

st.subheader("Chat with your documents using AI")

st.write(
    "Upload a PDF and ask questions in natural language. "
    "DocuMind AI retrieves relevant passages and answers "
    "with source page references."
)

st.divider()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        # Detect a changed file even if the filename is the same.
        import hashlib
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash != st.session_state.get("file_hash"):

            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.uploaded_filename = None
            st.session_state.file_hash = None

            temp_path = None

            try:
                with st.spinner(
                    "Reading PDF and creating knowledge base..."
                ):

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(file_bytes)
                        temp_path = temp_file.name

                    vector_store = process_pdf(temp_path)

                    st.session_state.vector_store = vector_store
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.file_hash = file_hash

                st.success("PDF processed successfully!")

            except Exception as e:
                st.error(f"PDF processing failed: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    if st.session_state.uploaded_filename:
        st.success(
            f"Loaded: {st.session_state.uploaded_filename}"
        )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------- MAIN CHAT ----------------

if st.session_state.vector_store is None:

    st.info("👈 Upload a PDF from the sidebar to get started.")

    st.markdown("""
    ### What can DocuMind AI do?

    - 📄 Read PDF documents.
    - 🔍 Retrieve relevant information.
    - 💬 Answer questions in natural language.
    - 📌 Show source page references.
    - 🧠 Summarize document information.

    ### How it works

    PDF → Text Extraction → Chunking → Embeddings
    → FAISS Retrieval → Groq LLM → Answer
    """)

else:

    st.subheader("💬 Ask your document")

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:

                    answer, sources = ask_question(
                        st.session_state.vector_store,
                        question
                    )

                    st.markdown(answer)

                    if sources:
                        source_text = ", ".join(
                            map(str, sources)
                        )

                        st.caption(
                            f"📌 Source pages: {source_text}"
                        )

                    else:
                        source_text = "None"

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (
                            answer
                            + f"\n\n📌 Source pages: {source_text}"
                        )
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
