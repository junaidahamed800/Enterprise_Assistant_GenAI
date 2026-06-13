from src_v2.p04_retriever import retrieve_context
from src_v2.p05_promptbuilder import build_prompt
from src_v2.p06_llm import generate_answer

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

    print("\n===== RETRIEVED CONTEXT =====")
    print(context)
    print("=============================\n")

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