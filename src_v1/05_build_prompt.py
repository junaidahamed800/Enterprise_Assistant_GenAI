from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to vector store
vector_store = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

# User query
query = "How many annual leaves are allowed?"

# Retrieve relevant chunks
results = vector_store.similarity_search(
    query,
    k=2
)

# Combine retrieved chunks into context
context = "\n".join([doc.page_content for doc in results])

# Build prompt
prompt = f"""
Use only the provided context.

If the answer is not found in the context,
say "Information not available."

Context:
{context}

Question:
{query}

Answer:
"""

print(prompt)