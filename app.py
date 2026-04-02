from evaluator import evaluate_response
import streamlit as st
from rag_pipeline import (
    load_and_chunk_pdf, create_vector_store, create_qa_chain, get_answer,
    generate_document_summary, build_bm25_index,
    create_comparison_chain, comparison_search, get_comparison_answer
)
from pdf_exporter import generate_pdf_report
from dotenv import load_dotenv
import tempfile
import os
from datetime import datetime, timedelta

load_dotenv()

# Lightweight Conversation Memory

class ConversationMemory:
    """Sliding-window conversation memory. Keeps the last k exchanges (human + AI pairs)."""
    def __init__(self, k=5):
        self.k = k
        self.exchanges = []  # list of (human_input, ai_output)

    def save_context(self, human_input, ai_output):
        self.exchanges.append((human_input, ai_output))
        if len(self.exchanges) > self.k:
            self.exchanges = self.exchanges[-self.k:]

    def load_history_string(self):
        if not self.exchanges:
            return ""
        lines = []
        for human, ai in self.exchanges:
            lines.append(f"Human: {human}")
            lines.append(f"AI: {ai[:300]}")
        return "\n".join(lines)

    @property
    def turn_count(self):
        return len(self.exchanges)

    def clear(self):
        self.exchanges = []

# Rate Limiter

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

# Medical Validator

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

# Feature 4: Confidence Grade

def get_confidence_grade(overall_pct, source_docs=None):
    """Returns (grade, emoji, explanation) with optional source page citation."""
    page_hint = ""
    if source_docs:
        first_page = source_docs[0].metadata.get('page', None) if source_docs else None
        if first_page is not None:
            page_hint = f" (page {int(first_page) + 1})"

    if overall_pct >= 90:
        return "A", "🟢", f"Very high confidence - answer fully supported by document{page_hint}"
    elif overall_pct >= 75:
        return "B", "🟡", f"High confidence - answer mostly supported by document{page_hint}"
    elif overall_pct >= 60:
        return "C", "🟠", f"Medium confidence - some claims may need verification"
    else:
        return "D", "🔴", f"Low confidence - treat this answer with caution"


def render_confidence_badge(grade, emoji, explanation):
    """Render a styled confidence pill badge via HTML."""
    colors = {"A": "#00c853", "B": "#ffd600", "C": "#ff9100", "D": "#ff5252"}
    bg = colors.get(grade, "#999")
    text_color = "#000" if grade in ("A", "B") else "#fff"

    st.markdown(f"""
    <div style="display:inline-flex; align-items:center; gap:8px; margin:8px 0;">
        <span style="
            background:{bg}; color:{text_color};
            font-weight:700; font-size:18px;
            padding:4px 14px; border-radius:20px;
            letter-spacing:1px;
        ">{emoji} Grade {grade}</span>
        <span style="color:#555; font-size:13px;">{explanation}</span>
    </div>
    """, unsafe_allow_html=True)


# Feature 1: Memory Helper

def get_memory_string():
    """Get formatted conversation history from memory."""
    if "memory" not in st.session_state or st.session_state.memory is None:
        return ""
    return st.session_state.memory.load_history_string()


def save_to_memory(question, answer):
    """Save a Q&A exchange to memory."""
    if "memory" in st.session_state and st.session_state.memory is not None:
        st.session_state.memory.save_context(question, answer)


# Page Config

st.set_page_config(page_title="MedQuery", page_icon="🏥", layout="centered")
st.title("🏥 MedQuery")
st.caption("Advanced RAG · Hybrid Search · Evaluation Metrics · Conversation Memory · Document Comparison")
st.warning("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

# Session State Init

defaults = {
    "prompt": None, "llm": None, "bm25": None,
    "chunks": None, "vector_store": None,
    "chat_history": [], "doc_verified": False,
    "doc_summary": None,
    # Feature 1: LangChain memory
    "memory": None,
    # Feature 2: Comparison mode
    "compare_mode": False,
    "vs_a": None, "vs_b": None,
    "bm25_a": None, "bm25_b": None,
    "chunks_a": None, "chunks_b": None,
    "comparison_prompt": None,
    "doc_summary_a": None, "doc_summary_b": None,
    "doc_names": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Feature 2: Mode Toggle

mode = st.radio(
    "Mode",
    ["📄 Single Document", "🔀 Compare Documents"],
    horizontal=True,
    key="mode_select"
)
is_compare = (mode == "🔀 Compare Documents")

# Feature 1: Memory Indicator

if st.session_state.memory is not None and st.session_state.memory.turn_count > 0:
    turns = st.session_state.memory.turn_count
    st.info(f"🧠 Conversation memory active - {turns} turn{'s' if turns != 1 else ''} stored (last 5 kept)")

# SINGLE DOCUMENT MODE

if not is_compare:

    uploaded_files = st.file_uploader(
        "Upload medical PDFs (lab reports, prescriptions, discharge summaries)",
        type="pdf",
        accept_multiple_files=True,
        key="single_upload"
    )

    if uploaded_files and st.session_state.prompt is None:
        with st.spinner("Analyzing and verifying documents..."):
            all_chunks = []
            doc_names = []
            for uploaded_file in uploaded_files:
                doc_names.append(uploaded_file.name)
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
            st.session_state.doc_names = doc_names

            # Feature 1: Initialize LangChain memory
            st.session_state.memory = ConversationMemory(k=5)

        with st.spinner("Generating document summary and extracting key findings..."):
            summary = generate_document_summary(all_chunks, llm)
            st.session_state.doc_summary = summary

        st.success(f"✅ {len(all_chunks)} chunks indexed from {len(uploaded_files)} document(s). Ready!")

    # Document Summary Panel

    if st.session_state.doc_summary:
        with st.expander("📋 Document Analysis - Key Findings & Summary", expanded=True):
            lines = st.session_state.doc_summary.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                elif line.startswith(('1.', '2.', '3.', '4.')):
                    st.markdown(f"**{line}**")
                elif line.startswith('⚠️'):
                    st.error(line)
                elif line.startswith('-'):
                    st.markdown(line)
                else:
                    st.write(line)

    # Chat History Display

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if "sources" in message:
                    with st.expander("View sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1} - Page {source['page']}**")
                            st.caption(source["content"])
                if "evaluation" in message:
                    ev = message["evaluation"]
                    src_docs = message.get("source_docs_raw", None)
                    grade, emoji, explanation = get_confidence_grade(ev['overall_pct'], src_docs)
                    render_confidence_badge(grade, emoji, explanation)
                    with st.expander("RAG Evaluation Metrics"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Faithfulness", f"{ev['faithfulness_pct']}%")
                        col2.metric("Relevance", f"{ev['answer_relevance_pct']}%")
                        col3.metric("Precision", f"{ev['retrieval_precision_pct']}%")
                        col4.metric("Utilisation", f"{ev['context_utilisation_pct']}%")

    # Chat Input

    if st.session_state.prompt:
        question = st.chat_input("Ask a medical question about your document...")

        if question:
            if not check_rate_limit():
                st.stop()

            # Feature 1 - Get conversation history from LangChain memory
            chat_history = get_memory_string()

            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("MedQuery analyzing with hybrid search + re-ranking..."):
                    answer, source_docs, rewritten_query = get_answer(
                        st.session_state.prompt,
                        st.session_state.llm,
                        st.session_state.bm25,
                        st.session_state.chunks,
                        st.session_state.vector_store,
                        question,
                        chat_history=chat_history
                    )
                    evaluation = evaluate_response(question, answer, source_docs)

                st.write(answer)

                # Feature 4 - Styled confidence badge with source page
                grade, emoji, explanation = get_confidence_grade(evaluation['overall_pct'], source_docs)
                render_confidence_badge(grade, emoji, explanation)

                with st.expander("Query rewriting"):
                    st.caption(f"Original: {question}")
                    st.caption(f"Rewritten: {rewritten_query}")

                sources = []
                with st.expander("View sources"):
                    for i, doc in enumerate(source_docs):
                        page = doc.metadata.get('page', 'N/A')
                        content = doc.page_content[:300]
                        st.markdown(f"**Source {i+1} - Page {page}**")
                        st.caption(content)
                        sources.append({"page": page, "content": content})

                with st.expander("RAG Evaluation Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Faithfulness", f"{evaluation['faithfulness_pct']}%")
                    col2.metric("Relevance", f"{evaluation['answer_relevance_pct']}%")
                    col3.metric("Precision", f"{evaluation['retrieval_precision_pct']}%")
                    col4.metric("Utilisation", f"{evaluation['context_utilisation_pct']}%")
                    overall = evaluation['overall_pct']
                    bar_color = "#00c853" if overall >= 70 else "#ffb300" if overall >= 40 else "#ff5252"
                    st.markdown(f"""
                    **Overall RAG Score: {overall}%**
                    <div style="background:#f0f0f0; border-radius:8px; height:10px; margin-top:4px;">
                        <div style="background:{bar_color}; width:{overall}%; height:10px; border-radius:8px;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Faithfulness · Relevance · Precision · Utilisation")

            # Feature 1 - Save to LangChain memory
            save_to_memory(question, answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "evaluation": evaluation,
                "source_docs_raw": source_docs
            })

    elif not uploaded_files:
        st.info("👆 Upload one or more medical PDFs above to get started.")


# FEATURE 2: DOCUMENT COMPARISON MODE

else:
    st.markdown("---")
    st.subheader("🔀 Document Comparison Mode")
    st.caption("Upload two medical PDFs and ask questions that compare them.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔵 Report A")
        file_a = st.file_uploader("Upload Report A", type="pdf", key="compare_a")

    with col_b:
        st.markdown("#### 🟠 Report B")
        file_b = st.file_uploader("Upload Report B", type="pdf", key="compare_b")

    # Process both files when both are uploaded
    if file_a and file_b and st.session_state.vs_a is None:
        with st.spinner("Processing and indexing both documents..."):
            # Process Report A
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_a.read())
                tmp_path_a = tmp.name
            chunks_a = load_and_chunk_pdf(tmp_path_a)
            os.unlink(tmp_path_a)

            # Process Report B
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_b.read())
                tmp_path_b = tmp.name
            chunks_b = load_and_chunk_pdf(tmp_path_b)
            os.unlink(tmp_path_b)

            # Validate both are medical
            if not is_medical_document(chunks_a):
                st.error("❌ Report A is not a medical document.")
                st.stop()
            if not is_medical_document(chunks_b):
                st.error("❌ Report B is not a medical document.")
                st.stop()

            # Create separate indexes
            vs_a = create_vector_store(chunks_a)
            vs_b = create_vector_store(chunks_b)
            bm25_a = build_bm25_index(chunks_a)
            bm25_b = build_bm25_index(chunks_b)

            # LLM + comparison prompt
            from rag_pipeline import create_qa_chain
            _, llm, _, _ = create_qa_chain(vs_a, chunks_a)
            comp_prompt = create_comparison_chain(llm)

            # Store everything
            st.session_state.vs_a = vs_a
            st.session_state.vs_b = vs_b
            st.session_state.bm25_a = bm25_a
            st.session_state.bm25_b = bm25_b
            st.session_state.chunks_a = chunks_a
            st.session_state.chunks_b = chunks_b
            st.session_state.llm = llm
            st.session_state.comparison_prompt = comp_prompt
            st.session_state.doc_names = [file_a.name, file_b.name]

            # Feature 1: Initialize memory for comparison mode too
            st.session_state.memory = ConversationMemory(k=5)

        # Generate summaries for both
        with st.spinner("Generating summaries for both documents..."):
            summary_a = generate_document_summary(chunks_a, llm)
            summary_b = generate_document_summary(chunks_b, llm)
            st.session_state.doc_summary_a = summary_a
            st.session_state.doc_summary_b = summary_b

        st.success(f"✅ Both documents indexed - Report A: {len(chunks_a)} chunks, Report B: {len(chunks_b)} chunks. Ready to compare!")

    # Show summaries side by side

    if st.session_state.doc_summary_a and st.session_state.doc_summary_b:
        col_sa, col_sb = st.columns(2)
        with col_sa:
            with st.expander("🔵 Report A - Summary", expanded=True):
                st.write(st.session_state.doc_summary_a)
        with col_sb:
            with st.expander("🟠 Report B - Summary", expanded=True):
                st.write(st.session_state.doc_summary_b)

    # Comparison Chat History

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if "sources_a" in message or "sources_b" in message:
                    with st.expander("🔵 Report A sources"):
                        for i, src in enumerate(message.get("sources_a", [])):
                            st.markdown(f"**Source {i+1} - Page {src['page']}**")
                            st.caption(src["content"])
                    with st.expander("🟠 Report B sources"):
                        for i, src in enumerate(message.get("sources_b", [])):
                            st.markdown(f"**Source {i+1} - Page {src['page']}**")
                            st.caption(src["content"])
                if "evaluation" in message:
                    ev = message["evaluation"]
                    grade, emoji, explanation = get_confidence_grade(ev['overall_pct'])
                    render_confidence_badge(grade, emoji, explanation)
                    with st.expander("RAG Evaluation Metrics"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Faithfulness", f"{ev['faithfulness_pct']}%")
                        col2.metric("Relevance", f"{ev['answer_relevance_pct']}%")
                        col3.metric("Precision", f"{ev['retrieval_precision_pct']}%")
                        col4.metric("Utilisation", f"{ev['context_utilisation_pct']}%")

    # Comparison Chat Input

    if st.session_state.comparison_prompt:
        question = st.chat_input("Compare the documents - e.g. 'How do the symptoms differ?'")

        if question:
            if not check_rate_limit():
                st.stop()

            chat_history = get_memory_string()

            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("MedQuery comparing both documents..."):
                    docs_a, docs_b, rewritten = comparison_search(
                        question,
                        st.session_state.vs_a, st.session_state.bm25_a, st.session_state.chunks_a,
                        st.session_state.vs_b, st.session_state.bm25_b, st.session_state.chunks_b,
                        st.session_state.llm
                    )

                    answer, all_sources, _, _ = get_comparison_answer(
                        st.session_state.comparison_prompt,
                        st.session_state.llm,
                        question,
                        docs_a, docs_b,
                        chat_history=chat_history
                    )

                    evaluation = evaluate_response(question, answer, all_sources)

                st.write(answer)

                # Feature 4 - Styled confidence badge
                grade, emoji, explanation = get_confidence_grade(evaluation['overall_pct'])
                render_confidence_badge(grade, emoji, explanation)

                with st.expander("Query rewriting"):
                    st.caption(f"Original: {question}")
                    st.caption(f"Rewritten: {rewritten}")

                sources_a = []
                with st.expander("🔵 Report A sources"):
                    for i, doc in enumerate(docs_a):
                        page = doc.metadata.get('page', 'N/A')
                        content = doc.page_content[:300]
                        st.markdown(f"**Source {i+1} - Page {page}**")
                        st.caption(content)
                        sources_a.append({"page": page, "content": content})

                sources_b = []
                with st.expander("🟠 Report B sources"):
                    for i, doc in enumerate(docs_b):
                        page = doc.metadata.get('page', 'N/A')
                        content = doc.page_content[:300]
                        st.markdown(f"**Source {i+1} - Page {page}**")
                        st.caption(content)
                        sources_b.append({"page": page, "content": content})

                with st.expander("RAG Evaluation Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Faithfulness", f"{evaluation['faithfulness_pct']}%")
                    col2.metric("Relevance", f"{evaluation['answer_relevance_pct']}%")
                    col3.metric("Precision", f"{evaluation['retrieval_precision_pct']}%")
                    col4.metric("Utilisation", f"{evaluation['context_utilisation_pct']}%")
                    overall = evaluation['overall_pct']
                    bar_color = "#00c853" if overall >= 70 else "#ffb300" if overall >= 40 else "#ff5252"
                    st.markdown(f"""
                    **Overall RAG Score: {overall}%**
                    <div style="background:#f0f0f0; border-radius:8px; height:10px; margin-top:4px;">
                        <div style="background:{bar_color}; width:{overall}%; height:10px; border-radius:8px;"></div>
                    </div>
                    """, unsafe_allow_html=True)

            # Feature 1 - Save to memory
            save_to_memory(question, answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources_a": sources_a,
                "sources_b": sources_b,
                "evaluation": evaluation
            })

    elif not (file_a and file_b):
        st.info("👆 Upload both Report A and Report B above to start comparing.")


# Bottom Controls

if st.session_state.prompt or st.session_state.comparison_prompt:
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear and start over"):
            for key in defaults:
                if key in ("chat_history", "doc_names"):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
            st.rerun()

    with col2:
        # Feature 5 - Export as formatted PDF
        if st.session_state.chat_history:
            try:
                pdf_bytes = generate_pdf_report(
                    st.session_state.chat_history,
                    doc_names=st.session_state.doc_names
                )
                st.download_button(
                    label="📄 Export as PDF Report",
                    data=pdf_bytes,
                    file_name=f"medquery_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF export failed: {e}")