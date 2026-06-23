import os
from dotenv import load_dotenv
from src_v2.p01_document_loader import load_documents
from src_v2.p02_chunker import create_chunks
from src_v2.p03_vectorestore import create_vectorstore
from pinecone import Pinecone
load_dotenv()

# Clear existing vectors before re-indexing
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("enterprise-assistant")

# Only delete if index has vectors
stats = index.describe_index_stats()
if stats.total_vector_count > 0:
    index.delete(delete_all=True)
    print("Cleared existing vectors!")
else:
    print("Index is empty, skipping clear step!")


# Indexing after clearing existing vectors.
print("Starting Indexing...")

documents = load_documents()
print("Documents Loaded Sucessfully!")
print(f"Total Documents: {len(documents)}")

chunks = create_chunks(documents)
print("Chunking Completed Successfully!")

create_vectorstore(chunks)
print("Embeddings Created & Stored in Chroma DB Successfully!")
print("Indexing Completed (p01 - p03)")