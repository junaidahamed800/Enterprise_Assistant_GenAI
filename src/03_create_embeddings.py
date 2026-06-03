from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the document
loader = TextLoader("data/company_policy.txt")
documents = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert chunks into embeddings and store in ChromaDB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vectorstore"
)

print("Embeddings created successfully.")
print("Vectors stored in ChromaDB.")