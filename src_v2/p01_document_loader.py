from langchain_community.document_loaders import TextLoader


def load_documents():

    loader = TextLoader(
        "data/company_policy.txt"
    )

    return loader.load()