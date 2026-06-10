from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# Configure Gemini API
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

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

# Retrieve context
results = vector_store.similarity_search(
    query,
    k=2
)

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

# Generate answer
response = model.generate_content(prompt)

print(response.text)