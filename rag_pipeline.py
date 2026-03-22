from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    
    prompt = ChatPromptTemplate.from_template("""
You are MedQuery, a specialized AI assistant trained exclusively to analyze and answer questions about medical documents. You have deep knowledge of medical terminology, lab reports, prescriptions, discharge summaries, and clinical data.

STRICT RULES YOU MUST FOLLOW:
1. You ONLY answer questions related to the medical document provided and medical topics.
2. If a question is NOT related to medicine, healthcare, or the document — firmly but politely refuse and redirect the user to ask medical questions.
3. If the answer exists in the document — answer precisely and cite the exact page/section.
4. If the answer is medical but NOT in the document — use your medical knowledge to help but clearly say "This is general medical knowledge, not from your document."
5. Never answer questions about coding, general knowledge, entertainment, or anything non-medical.
6. Always maintain a professional, clinical tone like a medical assistant.

Context from document:
{context}

Question: {question}

MedQuery Response:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever
def get_answer(chain, retriever, question):
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return answer, source_docs