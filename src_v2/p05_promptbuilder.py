def build_prompt(
    query,
    context
):

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

    return prompt