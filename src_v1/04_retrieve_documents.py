from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing vector store
vector_store = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

# User query
query = "How many annual leaves are allowed?"

# Retrieve top 3 similar chunks
results = vector_store.similarity_search(
    query,
    k=2
)

# Print retrieved chunks
for doc in results:
    print(doc.page_content)