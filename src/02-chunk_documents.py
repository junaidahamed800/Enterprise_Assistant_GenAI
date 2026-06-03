from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document as a LangChain Document object
loader = TextLoader("data/company_policy.txt")
documents = loader.load()


# Configure chunk size and overlap settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)


# Split the document into smaller chunks
chunks = text_splitter.split_documents(documents)


# Display total number of chunks created
print(f"\nTotal Chunks Created: {len(chunks)}")


# Print chunk content and metadata for verification
for i, chunk in enumerate(chunks):

    print("\n==============================")
    print(f"Chunk {i + 1}")

    print("\nContent:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)