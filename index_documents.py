from src_v2.p01_document_loader import load_documents
from src_v2.p02_chunker import create_chunks
from src_v2.p03_vectorestore import create_vectorstore

print("Starting Indexing...")

documents = load_documents()
print("Documents Loaded Sucessfully!")


print(f"Total Documents: {len(documents)}")

chunks = create_chunks(
    documents
)
print("Chunking Completed Successfully!")

create_vectorstore(
    chunks
)
print("Embeddings Created & Stored in Chroma DB Successfully!")
print("Indexing Completed (p01 - p03)")