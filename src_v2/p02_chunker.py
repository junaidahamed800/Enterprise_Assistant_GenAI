from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

print("Chunking the data...")
def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    print(f"Total Chunks Created: {len(chunks)}")

    return chunks