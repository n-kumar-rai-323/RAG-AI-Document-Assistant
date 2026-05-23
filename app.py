import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# INIT


load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title(" RAG PDF Chatbot (Pro UI)")


# SESSION STATE


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# SIDEBAR - UPLOAD ONLY ONCE


with st.sidebar:
    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader("Choose PDF", type="pdf")

    if uploaded_file and st.session_state.vectorstore is None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        split_docs = splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.success("Embeddings Created!")


# CHECK DB READY


if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF to start chatting")
    st.stop()


# LLM


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Answer only from context."),
    ("human",
     "Context:\n{context}\n\nQuestion:\n{question}")
])

retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)


# CHAT INPUT


user_query = st.chat_input("Ask something from your PDF...")


# PROCESS QUERY


if user_query:

    docs = retriever.invoke(user_query)

    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": user_query
    })

    response = llm.invoke(final_prompt)

    # Save chat history
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("ai", response.content))


# DISPLAY CHAT HISTORY


for role, message in st.session_state.chat_history:

    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)