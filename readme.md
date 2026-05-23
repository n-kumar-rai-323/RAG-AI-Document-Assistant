# 📚 RAG PDF Chatbot (Streamlit + LangChain + Groq)

A powerful **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload PDFs and ask questions based on their content. Built using **Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Groq LLM (Llama 3.1)**.

---

## 🚀 Features

- 📄 Upload any PDF document
- ✂️ Automatic text chunking using RecursiveCharacterTextSplitter
- 🧠 Semantic search using HuggingFace embeddings
- 📦 Vector storage with ChromaDB
- 🤖 AI responses using Groq (Llama 3.1)
- 💬 Chat-style UI with history support
- ⚡ Fast and lightweight inference

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (Llama 3.1 Instant)
- **Framework:** LangChain
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector DB:** ChromaDB
- **PDF Loader:** PyPDFLoader
- **Environment Management:** python-dotenv

---

## 📁 Project Structure
├── app.py
├── .env
├── requirements.txt
├── chroma_db/ # auto-created vector database
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository
### Install dependencies
uv pip install -r requirements.txt
### Set up environment variables
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key_here
### Run the App
streamlit run app.py
