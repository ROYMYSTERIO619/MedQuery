from evaluator import evaluate_response
import streamlit as st
from rag_pipeline import load_and_chunk_pdf, create_vector_store, create_qa_chain, get_answer
from dotenv import load_dotenv
import tempfile
import os
from datetime import datetime, timedelta

load_dotenv()

def check_rate_limit():
    now = datetime.now()
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    if "window_start" not in st.session_state:
        st.session_state.window_start = now
    if "daily_count" not in st.session_state:
        st.session_state.daily_count = 0
    if "daily_reset" not in st.session_state:
        st.session_state.daily_reset = now + timedelta(days=1)
    if now - st.session_state.window_start > timedelta(minutes=1):
        st.session_state.request_count = 0
        st.session_state.window_start = now
    if now > st.session_state.daily_reset:
        st.session_state.daily_count = 0
        st.session_state.daily_reset = now + timedelta(days=1)
    if st.session_state.request_count >= 5:
        wait = 60 - (now - st.session_state.window_start).seconds
        st.error(f"⏳ Rate limit reached. Please wait {wait} seconds.")
        return False
    if st.session_state.daily_count >= 20:
        st.error("📵 Daily limit of 20 questions reached. Come back tomorrow.")
        return False
    st.session_state.request_count += 1
    st.session_state.daily_count += 1
    return True

def is_medical_document(chunks):
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
    return matches >= 5

st.set_page_config(page_title="MedQuery", page_icon="🏥", layout="centered")
st.title("🏥 MedQuery")
st.caption("Your intelligent medical document assistant — Advanced RAG + Evaluation")
st.warning("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

if "prompt" not in st.session_state:
    st.session_state.prompt = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_verified" not in st.session_state:
    st.session_state.doc_verified = False

uploaded_files = st.file_uploader(
    "Upload medical PDFs (lab reports, prescriptions, discharge summaries)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.session_state.prompt is None:
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
            st.error("❌ Not a medical document. Please upload a valid medical PDF.")
            st.stop()

        vector_store = create_vector_store(all_chunks)
        prompt, llm, bm25, chunks = create_qa_chain(vector_store, all_chunks)
        st.session_state.prompt = prompt
        st.session_state.llm = llm
        st.session_state.bm25 = bm25
        st.session_state.chunks = chunks
        st.session_state.vector_store = vector_store
        st.session_state.doc_verified = True
        st.success(f"✅ {len(all_chunks)} chunks indexed from {len(uploaded_files)} document(s). Ready!")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} — Page {source['page']}**")
                    st.caption(source["content"])
            if "evaluation" in message:
                with st.expander("RAG Evaluation Metrics"):
                    ev = message["evaluation"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Faithfulness", f"{ev['faithfulness_pct']}%")
                    col2.metric("Answer Relevance", f"{ev['answer_relevance_pct']}%")
                    col3.metric("Retrieval Precision", f"{ev['retrieval_precision_pct']}%")
                    col4.metric("Context Utilisation", f"{ev['context_utilisation_pct']}%")

if st.session_state.prompt:
    question = st.chat_input("Ask a medical question about your document...")

    if question:
        if not check_rate_limit():
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("MedQuery is analyzing with hybrid search + re-ranking..."):
                answer, source_docs, rewritten_query = get_answer(
                    st.session_state.prompt,
                    st.session_state.llm,
                    st.session_state.bm25,
                    st.session_state.chunks,
                    st.session_state.vector_store,
                    question
                )
                evaluation = evaluate_response(question, answer, source_docs)

            st.write(answer)

            with st.expander("Query rewriting"):
                st.caption(f"Your question: {question}")
                st.caption(f"Rewritten by MedQuery: {rewritten_query}")

            sources = []
            with st.expander("View sources"):
                for i, doc in enumerate(source_docs):
                    page = doc.metadata.get('page', 'N/A')
                    content = doc.page_content[:300]
                    st.markdown(f"**Source {i+1} — Page {page}**")
                    st.caption(content)
                    sources.append({"page": page, "content": content})

            with st.expander("RAG Evaluation Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Faithfulness", f"{evaluation['faithfulness_pct']}%")
                col2.metric("Answer Relevance", f"{evaluation['answer_relevance_pct']}%")
                col3.metric("Retrieval Precision", f"{evaluation['retrieval_precision_pct']}%")
                col4.metric("Context Utilisation", f"{evaluation['context_utilisation_pct']}%")

                overall = evaluation['overall_pct']
                bar_color = "#00c853" if overall >= 70 else "#ffb300" if overall >= 40 else "#ff5252"
                st.markdown(f"""
                **Overall RAG Score: {overall}%**
                <div style="background:#f0f0f0; border-radius:8px; height:10px; margin-top:4px;">
                    <div style="background:{bar_color}; width:{overall}%; height:10px; border-radius:8px;"></div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Faithfulness: grounded in docs · Relevance: addresses question · Precision: useful chunks · Utilisation: context used")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "evaluation": evaluation
        })

elif not uploaded_files:
    st.info("👆 Upload one or more medical PDFs above to get started.")

if st.session_state.prompt:
    if st.button("🗑️ Clear and upload new documents"):
        for key in ["prompt", "llm", "bm25", "chunks", "vector_store", "chat_history", "doc_verified"]:
            st.session_state[key] = None if key != "chat_history" else []
        st.session_state.doc_verified = False
        st.rerun()