import os
import shutil
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ----------------------------
# Config
# ----------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
TEMP_DIR = "temp_files"

st.set_page_config(page_title="Simple RAG App", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def load_documents(files):
    documents = []
    os.makedirs(TEMP_DIR, exist_ok=True)

    for file in files:
        file_path = os.path.join(TEMP_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    vectordb.persist()
    return vectordb


def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant answering questions using ONLY the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely:"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# ----------------------------
# UI
# ----------------------------
st.title("📄 Simple RAG App (Modern LangChain + Chroma + Streamlit)")

st.sidebar.header("📤 Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("📚 Index Documents"):
    if not uploaded_files:
        st.warning("Please upload some documents.")
    else:
        with st.spinner("Loading documents..."):
            docs = load_documents(uploaded_files)

        with st.spinner("Splitting documents..."):
            chunks = split_documents(docs)

        with st.spinner("Creating vector store..."):
            create_vectorstore(chunks)

        st.success(f"✅ Indexed {len(chunks)} chunks into ChromaDB!")

# ----------------------------
# QA Section
# ----------------------------
st.header("💬 Ask questions about your documents")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not os.path.exists(CHROMA_DIR):
        st.error("❌ No index found. Please upload and index documents first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            vectordb = load_vectorstore()
            rag_chain, retriever = build_rag_chain(vectordb)

            answer = rag_chain.invoke(question)

            # Also fetch sources
            docs = retriever.invoke(question)

            st.subheader("✅ Answer")
            st.write(answer)

            st.subheader("📚 Sources")
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}**")
                st.write(doc.metadata)
                st.write(doc.page_content[:500] + "...")
