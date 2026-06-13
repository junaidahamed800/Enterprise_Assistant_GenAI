from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

def retrieve_context(query):

    results = vector_store.similarity_search(
        query,
        k=2
    )

    return "\n".join(
        [doc.page_content for doc in results]
    )