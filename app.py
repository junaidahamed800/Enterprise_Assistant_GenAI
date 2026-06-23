import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
    page_title="Enterprise AI Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise AI Assistant")
st.caption("Upload your documents and ask questions")

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
    st.header("Upload Files 🗐")

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

# ── Main: Q&A Interface ────────────────────────────────────────
st.subheader("💬 Ask a Question")

# WHY st.chat_input instead of st.text_input + button:
# st.chat_input submits when user presses Enter — just like a chat app.
# No button click needed. It also sticks to the bottom of the screen
# and clears automatically after each submission.
question = st.chat_input("e.g. How many days of annual leave do I get?")

if question:
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