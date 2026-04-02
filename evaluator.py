from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

# ─── Metric 1: Faithfulness ───────────────────────────

def evaluate_faithfulness(answer, source_docs, llm):
    """
    Measures if the answer is actually grounded in retrieved chunks.
    Score 0-10: 10 = fully grounded, 0 = hallucinated
    """
    context = "\n\n".join(doc.page_content for doc in source_docs)

    prompt = ChatPromptTemplate.from_template("""
You are evaluating faithfulness of a medical AI response.

If the answer starts with "⚠️ Not found in your document" — score it 7/10 by default
since it correctly admitted the document doesn't contain the answer.

Otherwise rate 0-10:
- 10: Every claim directly supported by context
- 7: Mostly supported with minor extrapolation  
- 4: Some unsupported claims
- 0: Contradicts or ignores context entirely

Return ONLY a number. Nothing else.

Context: {context}
Answer: {answer}

Faithfulness score:""")

    chain = prompt | llm | StrOutputParser()
    try:
        score_str = chain.invoke({"context": context[:2000], "answer": answer})
        score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_str)))
        return min(max(round(score, 1), 0), 10)
    except:
        return 0.0

# ─── Metric 2: Answer Relevance ──────────────────────

def evaluate_answer_relevance(question, answer):
    """
    Measures if the answer actually addresses the question.
    Uses cosine similarity between question and answer embeddings.
    Score 0-1: 1 = perfectly relevant, 0 = completely off-topic
    """
    try:
        q_embedding = embeddings.embed_query(question)
        a_embedding = embeddings.embed_query(answer)

        q_vec = np.array(q_embedding)
        a_vec = np.array(a_embedding)

        cosine_sim = np.dot(q_vec, a_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(a_vec)
        )
        return round(float(cosine_sim), 3)
    except:
        return 0.0

# ─── Metric 3: Retrieval Precision ───────────────────

def evaluate_retrieval_precision(question, source_docs, llm):
    """
    Measures what % of retrieved chunks are actually relevant
    to the question. Scores each chunk 0 or 1.
    """
    if not source_docs:
        return 0.0

    prompt = ChatPromptTemplate.from_template("""
Is this document chunk relevant to answering the question?
Return ONLY 1 (relevant) or 0 (not relevant). Nothing else.

Question: {question}
Chunk: {chunk}
Relevant (1/0):""")

    chain = prompt | llm | StrOutputParser()
    scores = []

    for doc in source_docs:
        try:
            result = chain.invoke({
                "question": question,
                "chunk": doc.page_content[:300]
            })
            score = 1 if '1' in result else 0
            scores.append(score)
        except:
            scores.append(0)

    return round(sum(scores) / len(scores), 2) if scores else 0.0

# ─── Metric 4: Context Utilisation ───────────────────

def evaluate_context_utilisation(answer, source_docs):
    """
    Measures how much of the retrieved context was
    actually used in generating the answer.
    Uses word overlap as a proxy metric.
    """
    if not source_docs:
        return 0.0

    answer_words = set(answer.lower().split())
    context_words = set()
    for doc in source_docs:
        context_words.update(doc.page_content.lower().split())

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does',
                  'did', 'will', 'would', 'could', 'should', 'may',
                  'might', 'must', 'shall', 'can', 'need', 'dare',
                  'ought', 'used', 'to', 'of', 'in', 'for', 'on',
                  'with', 'at', 'by', 'from', 'as', 'into', 'through',
                  'and', 'or', 'but', 'if', 'then', 'that', 'this',
                  'it', 'its', 'not', 'no', 'nor', 'so', 'yet'}

    answer_words -= stop_words
    context_words -= stop_words

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    utilisation = len(overlap) / len(answer_words)
    return round(min(utilisation, 1.0), 2)

# ─── Master evaluation function ──────────────────────

def evaluate_response(question, answer, source_docs):
    """
    Runs all 4 metrics and returns a complete evaluation report.
    Called after every answer generation.
    """
    llm = get_llm()

    faithfulness = evaluate_faithfulness(answer, source_docs, llm)
    relevance = evaluate_answer_relevance(question, answer)
    precision = evaluate_retrieval_precision(question, source_docs, llm)
    utilisation = evaluate_context_utilisation(answer, source_docs)

    # Overall score — weighted average
    overall = round(
        (faithfulness / 10 * 0.4) +
        (relevance * 0.3) +
        (precision * 0.2) +
        (utilisation * 0.1),
        3
    )

    return {
        'faithfulness': faithfulness,
        'faithfulness_pct': round(faithfulness * 10, 1),
        'answer_relevance': relevance,
        'answer_relevance_pct': round(relevance * 100, 1),
        'retrieval_precision': precision,
        'retrieval_precision_pct': round(precision * 100, 1),
        'context_utilisation': utilisation,
        'context_utilisation_pct': round(utilisation * 100, 1),
        'overall_score': overall,
        'overall_pct': round(overall * 100, 1)
    }