import os
import shutil
import tempfile
from typing import List

import streamlit as st

import chromadb
from chromadb.config import Settings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Simple RAG (Streamlit + Chroma)", layout="wide")
st.title("Simple RAG (Streamlit + Chroma)")

# Streamlit Cloud: /tmp is reliably writable during runtime
PERSIST_DIR = "/tmp/chroma_db"
COLLECTION_NAME = "rag_collection"


# -----------------------------
# Utilities
# -----------------------------
def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def load_uploaded_pdfs(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        # PyPDFLoader expects a filepath, so write to a temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        file_docs = loader.load()

        # Add filename to metadata
        for d in file_docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = f.name

        docs.extend(file_docs)

        # cleanup
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return docs


def load_uploaded_text(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")

        docs.append(Document(page_content=text, metadata={"source": f.name}))
    return docs


@st.cache_resource
def get_embeddings():
    # Requires OPENAI_API_KEY in Streamlit secrets or env
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource
def get_chroma_client() -> chromadb.Client:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    # PersistentClient forces embedded local mode (avoids tenant/default_tenant server validation issues)
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def reset_vectorstore():
    # Clear cached resources
    get_vectorstore.clear()
    get_chroma_client.clear()
    get_embeddings.clear()

    # Delete persisted data
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)


def format_docs(docs: List[Document]) -> str:
    # Keep it readable; include source markers
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src}" + (f" (page {page})" if page is not None else "")
        parts.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    k = st.slider("Top-K chunks", min_value=2, max_value=10, value=4, step=1)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.1)

    if st.button("🧹 Reset vector DB (delete persisted data)"):
        reset_vectorstore()
        st.success("Vector DB reset. Re-upload files and rebuild.")

    st.caption(f"Persist dir: `{PERSIST_DIR}`")
    st.caption(f"Collection: `{COLLECTION_NAME}`")


# -----------------------------
# Upload + Build
# -----------------------------
st.subheader("1) Upload documents")
pdf_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

text_files = st.file_uploader(
    "Upload text files (txt/md/csv/log)",
    type=["txt", "md", "csv", "log"],
    accept_multiple_files=True,
)

build_clicked = st.button("📥 Build / Update Vector Store", type="primary")

if build_clicked:
    if not pdf_files and not text_files:
        st.warning("Upload at least one PDF or text file first.")
    else:
        with st.spinner("Loading files..."):
            docs: List[Document] = []
            if pdf_files:
                docs.extend(load_uploaded_pdfs(pdf_files))
            if text_files:
                docs.extend(load_uploaded_text(text_files))

        with st.spinner("Chunking..."):
            chunks = split_docs(docs)

        with st.spinner("Embedding & storing into Chroma..."):
            vs = get_vectorstore()
            vs.add_documents(chunks)

        st.success(f"Done. Added {len(chunks)} chunks.")


# -----------------------------
# Ask
# -----------------------------
st.subheader("2) Ask a question")
question = st.text_input("Question", placeholder="Ask something about your uploaded docs...")

if question:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say you don't know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Retrieving + generating answer..."):
        answer = rag_chain.invoke(question)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Show retrieved chunks"):
        retrieved = retriever.get_relevant_documents(question)
        for i, d in enumerate(retrieved, start=1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            st.markdown(f"**#{i} — {src}" + (f" (page {page})" if page is not None else "") + "**")
            st.write(d.page_content)
            st.divider()
