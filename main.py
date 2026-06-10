from src_v2.p01_document_loader import (
    load_documents
)

from src_v2.p02_chunker import (
    create_chunks
)

from src_v2.p03_vectorestore import (
    create_vectorstore
)

from src_v2.p04_retriever import (
    retrieve_context
)

from src_v2.p05_promptbuilder import (
    build_prompt
)

from src_v2.p06_llm import (
    generate_answer
)

# One-time setup

documents = load_documents()

chunks = create_chunks(
    documents
)

create_vectorstore(
    chunks
)

print("Enterprise Assistant RAG Chatbot")
print("Type 'exit' to quit.\n")

while True:

    query = input(
        "Ask Question: "
    )

    if query.lower() == "exit":
        break

    context = retrieve_context(
        query
    )

    prompt = build_prompt(
        query,
        context
    )

    answer = generate_answer(
        prompt
    )

    print("\nAnswer:")
    print(answer)
    print()