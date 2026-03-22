 
import streamlit as st
from rag_pipeline import load_and_chunk_pdf, create_vector_store, create_qa_chain, get_answer
import tempfile
import os

st.set_page_config(page_title="MedQuery", page_icon="🏥", layout="centered")

st.title("🏥 MedQuery")
st.caption("Ask anything from your medical documents")

st.warning("⚠️ This tool is for informational purposes only and is not a substitute for professional medical advice.")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload your medical PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.session_state.chain is None:
    with st.spinner("Processing your documents... this may take a minute"):
        all_chunks = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            chunks = load_and_chunk_pdf(tmp_path)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)

        vector_store = create_vector_store(all_chunks)
        chain, retriever = create_qa_chain(vector_store)
        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.success(f"Ready! Processed {len(all_chunks)} chunks from {len(uploaded_files)} document(s).")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.chain:
    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source_docs = get_answer(
                    st.session_state.chain,
                    st.session_state.retriever,
                    question
                )
            st.write(answer)

            with st.expander("View sources"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Source {i+1} — Page {doc.metadata.get('page', 'N/A')}**")
                    st.caption(doc.page_content[:300])

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Upload one or more PDF documents above to get started.")