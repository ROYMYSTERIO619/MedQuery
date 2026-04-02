from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

# ─── Step 1: Smart contextual chunking ────────────────
def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
        chunk.metadata['char_count'] = len(chunk.page_content)

    return chunks


# ─── Step 2: BM25 keyword index ───────────────────────
def build_bm25_index(chunks):
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


# ─── Step 3: Hybrid search with RRF ──────────────────
def reciprocal_rank_fusion(dense_docs, sparse_docs, k=60):
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def hybrid_search(query, vector_store, bm25, chunks, top_k=6):
    dense_results = vector_store.similarity_search(query, k=top_k)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    sparse_results = [chunks[i] for i in top_bm25_indices]

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]


# ─── Step 4: Query rewriting ─────────────────────────
def rewrite_query(question, llm):
    rewrite_prompt = ChatPromptTemplate.from_template("""
You are a medical search query optimizer.
Rewrite the following question into a precise medical search query.
Make it more specific using medical terminology where appropriate.
Return ONLY the rewritten query, nothing else.

Original question: {question}
Rewritten query:""")

    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    return rewritten.strip()


# ─── Step 5: Cross-encoder re-ranking ────────────────
def rerank_chunks(query, chunks, llm, top_n=3):
    if not chunks:
        return chunks

    rerank_prompt = ChatPromptTemplate.from_template("""
Rate how relevant this document chunk is to answering the medical question.
Return ONLY a number between 0 and 10. Nothing else.

Question: {question}
Document chunk: {chunk}
Relevance score (0-10):""")

    chain = rerank_prompt | llm | StrOutputParser()
    scored = []

    for chunk in chunks[:6]:
        try:
            score_str = chain.invoke({
                "question": query,
                "chunk": chunk.page_content[:300]
            })
            score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_str)))
            scored.append((score, chunk))
        except:
            scored.append((0, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]


# ─── Step 6: Vector store creation ───────────────────
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ─── Step 7: QA chain with full advanced pipeline ────
def create_qa_chain(vector_store, chunks):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

    bm25 = build_bm25_index(chunks)

    prompt = ChatPromptTemplate.from_template("""
You are MedQuery, a specialized medical document assistant with deep clinical knowledge.

RESPONSE RULES:
1. First check if the answer exists in the context below.
2. If YES — answer precisely from the document and cite the page/section.
3. If NO — you MAY use general medical knowledge BUT you MUST:
   - Start with: "Not found in your document. Based on general medical knowledge:"
   - Give accurate, evidence-based medical information
   - End with: "Always consult a qualified physician before making medical decisions."
4. For medicine recommendations specifically:
   - You CAN suggest common medications for conditions mentioned in the document
   - Always mention dosage ranges are general and must be confirmed by a doctor
   - Never recommend specific brands — use generic drug names only
5. NEVER answer non-medical questions under any circumstance.
6. Maintain professional, clinical tone always.
7. If there is previous conversation, use it to understand follow-up questions.

Previous conversation:
{chat_history}

Context from document:
{context}

Question: {question}

MedQuery Response:
""")

    return prompt, llm, bm25, chunks


def self_check_answer(answer, context, llm):
    check_prompt = ChatPromptTemplate.from_template("""
Review this answer against the provided context.
Identify any claims NOT supported by the context and remove or correct them.
Return the corrected answer only. If the answer is fully grounded, return it unchanged.

Context: {context}
Answer to verify: {answer}

Verified answer:""")

    chain = check_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "answer": answer})


def get_answer(prompt, llm, bm25, chunks, vector_store, question, chat_history=""):
    rewritten = rewrite_query(question, llm)

    candidates = hybrid_search(rewritten, vector_store, bm25, chunks, top_k=8)
    final_chunks = rerank_chunks(question, candidates, llm, top_n=4)

    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in final_chunks
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    # self-check applied properly
    answer = self_check_answer(answer, context, llm)

    return answer, final_chunks, rewritten
def generate_document_summary(chunks, llm):
    """
    Auto-generates executive summary + key findings + abnormal value flags
    when a document is first uploaded.
    """
    # Take first 15 chunks for summary — covers most of the key content
    sample_text = "\n\n".join(
        chunk.page_content for chunk in chunks[:15]
    )

    summary_prompt = ChatPromptTemplate.from_template("""
You are a medical document analyst. Analyze this medical document and provide:

1. EXECUTIVE SUMMARY: A precise 3-sentence summary of what this document is about.

2. KEY FINDINGS: Extract 5-8 key medical findings as bullet points.
   Format exactly like this:
   - Finding 1
   - Finding 2

3. ABNORMAL VALUES: List any values that appear abnormal or concerning.
   Format exactly like this:
   - ⚠️ [value name]: [value] — [why it's concerning]
   If no abnormal values found, write: "No abnormal values detected."

4. DOCUMENT TYPE: Identify what type of medical document this is
   (e.g. Lab Report, Discharge Summary, Prescription, Clinical Research Paper)

Medical document content:
{text}

Analysis:""")

    chain = summary_prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({"text": sample_text[:4000]})
        return summary
    except Exception as e:
        return f"Summary generation failed: {e}"


# ─── Feature 2: Document Comparison Mode ─────────────

def create_comparison_chain(llm):
    """Create a specialized prompt for comparing two medical documents."""
    comparison_prompt = ChatPromptTemplate.from_template("""
You are MedQuery, a specialized medical document comparison assistant.

You have been given content from TWO separate medical documents.
Your job is to compare and contrast them based on the user's question.

RESPONSE RULES:
1. Always clearly label which document you are referencing: [Report A] or [Report B].
2. When comparing values, present them side by side.
3. Highlight key differences and similarities.
4. Flag any medically significant discrepancies between the documents.
5. NEVER answer non-medical questions under any circumstance.
6. Maintain professional, clinical tone always.
7. If there is previous conversation, use it to understand follow-up questions.

Previous conversation:
{chat_history}

--- REPORT A ---
{context_a}

--- REPORT B ---
{context_b}

Question: {question}

MedQuery Comparison Response:
""")
    return comparison_prompt


def comparison_search(query, vs_a, bm25_a, chunks_a, vs_b, bm25_b, chunks_b, llm, top_k=4):
    """
    Search both document indexes and return tagged results from each.
    Uses the same hybrid search + reranking pipeline per document.
    """
    rewritten = rewrite_query(query, llm)

    # Search Report A
    candidates_a = hybrid_search(rewritten, vs_a, bm25_a, chunks_a, top_k=6)
    final_a = rerank_chunks(query, candidates_a, llm, top_n=top_k)

    # Search Report B
    candidates_b = hybrid_search(rewritten, vs_b, bm25_b, chunks_b, top_k=6)
    final_b = rerank_chunks(query, candidates_b, llm, top_n=top_k)

    return final_a, final_b, rewritten


def get_comparison_answer(comparison_prompt, llm, question, docs_a, docs_b, chat_history=""):
    """
    Generate an answer comparing content from two documents.
    """
    context_a = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs_a
    )
    context_b = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs_b
    )

    chain = comparison_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context_a": context_a,
        "context_b": context_b,
        "question": question,
        "chat_history": chat_history
    })

    # Self-check against merged context
    merged_context = f"REPORT A:\n{context_a}\n\nREPORT B:\n{context_b}"
    answer = self_check_answer(answer, merged_context, llm)

    # Merge source docs for evaluation (tag them)
    all_sources = docs_a + docs_b
    return answer, all_sources, docs_a, docs_b