# eval_runner.py
# Runs every question in evaluationSet through your RAG pipeline
# and prints Expected vs Actual side by side for manual review.

import sys
import os

sys.path.insert(0, os.path.abspath(".."))
from evalSet import evaluationSet
from src_v2.p04_retriever import retrieve_context
from src_v2.p05_promptbuilder import build_prompt
from src_v2.p06_llm import generate_answer

# ── The RAG pipeline as a single callable ─────────────────────
# WHY we wrap it in one function:
# The eval loop just needs to call rag_pipeline(question) → answer.
# Keeping retrieval + prompt + LLM together here makes the
# runner clean and easy to swap later (e.g. swap Pinecone in).

def rag_pipeline(question: str) -> str:

    # Step 1: Retrieve relevant chunks from vector store
    docs = retrieve_context(question)

    # Step 2: Build the prompt with retrieved context
    prompt = build_prompt(question, docs)

    # Step 3: Generate answer via LLM
    answer = generate_answer(prompt)

    return answer, docs  # return docs too so we can show retrieval


# ── Eval Runner ───────────────────────────────────────────────
def run_eval():

    print("=" * 60)
    print("  RAG EVALUATION REPORT")
    print("=" * 60)

    # Track counts per type
    results_by_type = {}

    for item in evaluationSet:

        question    = item["question"]
        expected    = item["expected_answer"]
        item_id     = item["id"]
        item_type   = item["type"]
        item_source = item["source"]

        # Run question through your full RAG pipeline
        actual_answer, retrieved_docs = rag_pipeline(question)

        # Print result block
        print(f"\n[{item_id}] TYPE: {item_type.upper()}  |  SOURCE: {item_source}")
        print(f"  Q        : {question}")
        print(f"  Expected : {expected}")
        print(f"  Got      : {actual_answer.strip()}")

        # Show which chunks were retrieved
        print(f"  Retrieved chunks ({len(retrieved_docs)}):")
        for i, doc in enumerate(retrieved_docs, 1):
            source_file = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"    Chunk {i} [{source_file}]: {preview}...")

        print("-" * 60)

        # Group by type for summary
        if item_type not in results_by_type:
            results_by_type[item_type] = 0
        results_by_type[item_type] += 1

    # Summary at the end
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total questions run : {len(evaluationSet)}")
    for qtype, count in results_by_type.items():
        print(f"  {qtype:<15} : {count} questions")
    print("\n  Review the 'Got' answers above manually.")
    print("  Look for: wrong answers, missing info, hallucinations.")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()