from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Creating Embeddings for the chunks & Storing in Chroma DB....")
def create_vectorstore(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="vectorstore"
    )
    return vector_store