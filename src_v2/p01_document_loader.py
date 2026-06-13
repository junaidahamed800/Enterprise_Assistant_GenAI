from langchain_community.document_loaders import TextLoader

print("Loading Documents...")
def load_documents():

    loader = TextLoader(
        "data/company_policy.txt"
    )

    return loader.load()