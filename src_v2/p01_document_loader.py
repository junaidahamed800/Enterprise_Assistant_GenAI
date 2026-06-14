from langchain_community.document_loaders import TextLoader
import os

def load_documents():

    documents = []

    for file in os.listdir("data"):

        if file.endswith(".txt"):

            loader = TextLoader(f"data/{file}")

            documents.extend(loader.load())

    return documents