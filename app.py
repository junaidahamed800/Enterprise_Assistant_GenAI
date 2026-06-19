import streamlit as st

from src_v2.p04_retriever import retrieve_context
from src_v2.p05_promptbuilder import build_prompt
from src_v2.p06_llm import generate_answer
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import your existing pipeline modules
# Adjust the import path to match your project structure:
# If running from project root with src_v2/ folder:
#   from src_v2.p01_document_loader import load_single_file
#   from src_v2.p02_chunker import create_chunks
#   etc.
# If using flat structure (all files in root), use direct imports:
from src_v2.p01_document_loader import load_single_file
from src_v2.p02_chunker import create_chunks
from src_v2.p04_retriever import retrieve_context
from src_v2.p05_promptbuilder import build_prompt
from src_v2.p06_llm import generate_answer

# ── Constants ──────────────────────────────────────────────────
DATA_DIR = "data"
VECTORSTORE_DIR = "chroma_db"
SUPPORTED_TYPES = ["txt", "pdf", "docx", "xlsx"]

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise Assistant")
st.caption("Upload documents and ask questions — powered by RAG")

# ── Helper: Save uploaded file to disk ─────────────────────────
def save_uploaded_file(uploaded_file) -> str:
    """
    Streamlit gives us an in-memory file object.
    We save it to the data/ folder on disk so our loader
    can process it the same way as manually placed files.

    WHY: LangChain loaders like PyPDFLoader and Docx2txtLoader
    need a real file path — they can't work with in-memory bytes.
    Saving first keeps our loader logic clean and reusable.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path


# ── Helper: Index a single file into ChromaDB ─────────────────
def index_file(filepath: str):
    """
    Loads → Chunks → Embeds → Stores one file into ChromaDB.

    WHY we index per file (not all files every time):
    Re-indexing everything on each upload is slow and wasteful.
    ChromaDB lets us ADD new documents to an existing collection,
    so we only process the newly uploaded file.
    """
    docs = load_single_file(filepath)

    if not docs:
        return 0

    chunks = create_chunks(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Chroma.from_documents() on an existing persist_directory
    # ADDS to the collection rather than overwriting it.
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTORSTORE_DIR,
    )

    return len(chunks)


# ── Sidebar: File Upload ───────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    st.write(
        "Upload any supported file and it will be indexed "
        "automatically into the knowledge base."
    )

    uploaded_files = st.file_uploader(
        label="Choose files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, XLSX, TXT",
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Files", use_container_width=True):

            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        filepath = save_uploaded_file(uploaded_file)
                        chunk_count = index_file(filepath)
                        st.success(
                            f"✅ {uploaded_file.name} — {chunk_count} chunks indexed"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to index {uploaded_file.name}: {e}")

    st.divider()

    # Show what's currently in the data folder
    st.subheader("📂 Indexed Files")
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        if files:
            for f in sorted(files):
                ext = os.path.splitext(f)[1].upper().lstrip(".")
                st.write(f"• `{f}` ({ext})")
        else:
            st.caption("No files yet. Upload some above!")
    else:
        st.caption("data/ folder not found.")


# ── Main: Q&A Interface ────────────────────────────────────────
st.subheader("💬 Ask a Question")

question = st.text_input(
    label="Type your question here:",
    placeholder="e.g. How many days of annual leave do I get?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Ask", use_container_width=True)

if submit:

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        with st.spinner("Searching knowledge base..."):

            # Step 1: Retrieve relevant chunks
            docs = retrieve_context(question)

            # Step 2: Build prompt with context
            prompt = build_prompt(question, docs)

            # Step 3: Generate answer via LLM
            answer = generate_answer(prompt)

        # Display answer
        st.subheader("📋 Answer")
        st.write(answer)

        # Show retrieved context (collapsed by default)
        with st.expander("🔎 Retrieved Context (chunks used to answer)"):
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                st.markdown(f"**Chunk {i}** — `{os.path.basename(source)}`")
                st.write(doc.page_content)
                st.divider()
st.set_page_config(
    page_title="Enterprise Assistant",
    page_icon="🤖",
)

st.title("🤖 Enterprise Assistant")

question = st.text_input(
    "Ask a question about company policies:"
)

if st.button("Submit"):

    if question:

        docs = retrieve_context(question)

        prompt = build_prompt(question, docs)

        answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for doc in docs:
                st.write(doc.page_content)