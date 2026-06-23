import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def retrieve_context(query):

    # Connect to existing Pinecone index
    vector_store = PineconeVectorStore(
        index_name="enterprise-assistant",
        embedding=embedding_model
    )

    results = vector_store.similarity_search(
        query,
        k=4
    )

    return results