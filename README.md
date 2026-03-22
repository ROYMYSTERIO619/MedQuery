# 🏥 MedQuery  
### *RAG-Powered Medical Document Assistant*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue"/>
  <img src="https://img.shields.io/badge/LangChain-1.2-green"/>
  <img src="https://img.shields.io/badge/Streamlit-1.55-red"/>
  <img src="https://img.shields.io/badge/Groq-Llama3.3-orange"/>
  <img src="https://img.shields.io/badge/FAISS-VectorSearch-purple"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow"/>
</p>

<p align="center">
  <b>Upload medical documents → Ask questions → Get cited answers instantly</b>
</p>

<p align="center">
  🔗 <a href="https://medqueryy.streamlit.app"><b>Live Demo</b></a>
</p>

---

## ✨ Why MedQuery?

> ⚡ A **safe, grounded, and explainable AI assistant** for medical documents

- 🧠 Answers only from **your uploaded documents**
- 📎 Shows **exact page citations**
- 🚫 **Zero hallucination** (says *“not found”* if missing)
- 📚 Works across **multiple PDFs**

---

## 📌 Overview

MedQuery is built using **Retrieval-Augmented Generation (RAG)**.

Upload:
- Lab reports  
- Prescriptions  
- Discharge summaries  
- Research papers  

Then ask questions in **natural language** and get **accurate, traceable answers**.

---

## 🚀 Features

| Feature | Description |
|--------|------------|
| 📄 PDF Parsing | Extracts medical text using PyPDF |
| 🔍 Smart Chunking | 500 tokens + 50 overlap |
| ⚡ FAISS Search | Fast local vector retrieval |
| 🧠 LLM (Groq) | Llama 3.3 for instant responses |
| 📎 Citations | Every answer is verifiable |
| 🛡️ Safety Layer | Prevents hallucination + misuse |
| ⏱️ Rate Limiting | Protects API usage |
| 📁 Multi-doc Support | Query across multiple PDFs |
| 💬 Memory | Follow-up questions supported |

---

## 🧠 How It Works

```
PDF → Text → Chunks → Embeddings → FAISS → Retrieval → LLM → Answer + Citations
```

---

## 🛠️ Tech Stack

| Layer        | Technology |
|-------------|-----------|
| LLM         | Llama 3.3 (Groq API) |
| Embeddings  | MiniLM-L6-v2 |
| Vector DB   | FAISS |
| Framework   | LangChain |
| Frontend    | Streamlit |
| Parsing     | PyPDF |

---

## ⚙️ Run Locally

### 1. Clone repository
```bash
git clone https://github.com/ROYMYSTERIO619/MedQuery
cd medquery
```

### 2. Setup environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 3. Add API Key
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
medquery/
├── app.py              # UI + validation
├── rag_pipeline.py     # Core RAG logic
├── requirements.txt
├── .env
└── README.md
```

---

## 🎯 Interview Edge

💡 **Why this project stands out:**
- Real-world use case (medical AI)
- Full RAG pipeline (not toy project)
- Safety + hallucination control
- Explainability via citations

---

## ⚠️ Limitations

- No OCR (scanned PDFs unsupported)  
- No image/table parsing  
- FAISS is not persistent  
- Large docs → slower initial processing  

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**  
> Not a substitute for professional medical advice

---

## 👨‍💻 Author

**Deepta Roy**

- 🔗 GitHub: https://github.com/ROYMYSTERIO619  
- 🔗 LinkedIn: https://www.linkedin.com/in/deepta-roy-2601852a1/  
- 🚀 Demo: https://medqueryy.streamlit.app  

---

<p align="center">
  ⭐ If you found this useful, consider starring the repo!
</p>
