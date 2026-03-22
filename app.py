import streamlit as st
from rag_pipeline import load_and_chunk_pdf, create_vector_store, create_qa_chain, get_answer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

def is_medical_document(chunks):
    # Strong medical keywords that won't appear in portfolios/resumes
    strong_medical_keywords = [
        "diagnosis", "prescription", "dosage", "symptom", "pathology",
        "radiology", "discharge summary", "lab report", "blood pressure",
        "hemoglobin", "cholesterol", "glucose", "platelet", "biopsy",
        "ecg", "mri", "ct scan", "x-ray", "icd", "medication", "mg",
        "mmhg", "clinical trial", "patient history", "chief complaint",
        "vital signs", "prognosis", "contraindication", "adverse effect",
        "hospitalization", "ward", "icu", "specimen", "serum", "plasma",
        "covid", "infection", "inflammatory", "antibiotic", "vaccine"
    ]
    
    full_text = " ".join([c.page_content.lower() for c in chunks[:20]])
    matches = sum(1 for word in strong_medical_keywords if word in full_text)
    
    # Needs at least 5 strong medical keyword matches
    return matches >= 5

st.set_page_config(
    page_title="MedQuery",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 MedQuery")
st.caption("Your intelligent medical document assistant")
st.warning("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_verified" not in st.session_state:
    st.session_state.doc_verified = False

uploaded_files = st.file_uploader(
    "Upload medical PDFs (lab reports, prescriptions, discharge summaries)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.session_state.chain is None:
    with st.spinner("Analyzing and verifying documents..."):
        all_chunks = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            chunks = load_and_chunk_pdf(tmp_path)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)

        if not is_medical_document(all_chunks):
            st.error("❌ This does not appear to be a medical document. MedQuery only accepts medical PDFs such as lab reports, prescriptions, discharge summaries, or clinical research papers. Please upload a valid medical document.")
            st.stop()

        vector_store = create_vector_store(all_chunks)
        chain, retriever = create_qa_chain(vector_store)
        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.session_state.doc_verified = True
        st.success(f"✅ Medical document verified and loaded. {len(all_chunks)} chunks indexed across {len(uploaded_files)} document(s). Ready for your questions.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} — Page {source['page']}**")
                    st.caption(source["content"])

if st.session_state.chain:
    question = st.chat_input("Ask a medical question about your document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("MedQuery is analyzing..."):
                answer, source_docs = get_answer(
                    st.session_state.chain,
                    st.session_state.retriever,
                    question
                )
            st.write(answer)

            sources = []
            with st.expander("View sources"):
                for i, doc in enumerate(source_docs):
                    page = doc.metadata.get('page', 'N/A')
                    content = doc.page_content[:300]
                    st.markdown(f"**Source {i+1} — Page {page}**")
                    st.caption(content)
                    sources.append({"page": page, "content": content})

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

elif not uploaded_files:
    st.info("👆 Upload one or more medical PDFs above to get started.")

if st.session_state.chain:
    if st.button("🗑️ Clear and upload new documents"):
        st.session_state.chain = None
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.doc_verified = False
        st.rerun()