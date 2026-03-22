from rag_pipeline import load_and_chunk_pdf, create_vector_store, create_qa_chain, get_answer

print("Step 1: Loading and chunking PDF...")
chunks = load_and_chunk_pdf('test.pdf')
print(f"Total chunks: {len(chunks)}")

print("\nStep 2: Creating vector store (this will take 1-2 mins first time)...")
vector_store = create_vector_store(chunks)
print("Vector store ready!")

print("\nStep 3: Creating QA chain...")
chain, retriever = create_qa_chain(vector_store)
print("Chain ready!")

print("\nStep 4: Asking a question...")
question = "What is this document about?"
answer, source_docs = get_answer(chain, retriever, question)

print(f"\nQuestion: {question}")
print(f"\nAnswer: {answer}")
print(f"\nSources used: {len(source_docs)} chunks")
for i, doc in enumerate(source_docs):
    print(f"\n--- Source {i+1} (Page {doc.metadata.get('page', 'N/A')}) ---")
    print(doc.page_content[:200])