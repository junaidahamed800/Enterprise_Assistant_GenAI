from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )

    return splitter.split_documents(documents)