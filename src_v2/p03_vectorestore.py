import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def create_vectorstore(chunks):

    # Step 1 — Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Step 2 — Connect to your existing index
    index = pc.Index("enterprise-assistant")

    # Step 3 — Same embedding model as before
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4 — Store chunks in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name="enterprise-assistant"
    )

    return vector_store