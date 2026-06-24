"""
RAG Evaluation Set — Apex Solutions Policy Documents
Use this to test your pipeline's retrieval + answer quality.

Format:
  - question: what you'd ask the RAG system
  - expected_answer: the correct answer (ground truth)
  - source: which document contains the answer
  - type: factual | edge_case | out_of_scope
"""

evaluationSet = [

    # ── LEAVE POLICY ──────────────────────────────────────────────
    {
        "id": "L01",
        "question": "How many days of annual leave does a permanent employee get per year?",
        "expected_answer": "18 days per calendar year, accruing at 1.5 days per month.",
        "source": "leave_policy",
        "type": "factual",
    },
    {
        "id": "L02",
        "question": "Can unused casual leave be carried forward to next year?",
        "expected_answer": "No. Casual leave cannot be carried forward or encashed at year end.",
        "source": "leave_policy",
        "type": "factual",
    },
    {
        "id": "L03",
        "question": "How much maternity leave is a female employee entitled to for her third child?",
        "expected_answer": "12 weeks. The 26-week entitlement applies only for the first two deliveries.",
        "source": "leave_policy",
        "type": "edge_case",
    },
    {
        "id": "L04",
        "question": "Within how many days must compensatory leave be claimed?",
        "expected_answer": "Within 60 days of the date it was earned. Unclaimed comp-off beyond this lapses.",
        "source": "leave_policy",
        "type": "factual",
    },
    {
        "id": "L05",
        "question": "Can an employee take 4 consecutive days of casual leave?",
        "expected_answer": "No. Casual leave is capped at 2 consecutive days. For longer absences, annual leave must be applied.",
        "source": "leave_policy",
        "type": "edge_case",
    },

    # ── HR POLICY ─────────────────────────────────────────────────
    {
        "id": "H01",
        "question": "What is the notice period for junior and mid-level employees?",
        "expected_answer": "One month. Senior management roles require three months.",
        "source": "hr_policy",
        "type": "factual",
    },
    {
        "id": "H02",
        "question": "How long does a Performance Improvement Plan (PIP) typically run?",
        "expected_answer": "60 to 90 days.",
        "source": "hr_policy",
        "type": "factual",
    },
    {
        "id": "H03",
        "question": "What are the standard working hours at Apex Solutions?",
        "expected_answer": "9:00 AM to 6:00 PM, Monday through Friday, with a one-hour lunch break.",
        "source": "hr_policy",
        "type": "factual",
    },
    {
        "id": "H04",
        "question": "How many free counseling sessions does the EAP provide per year?",
        "expected_answer": "Up to 6 free counseling sessions per year per employee.",
        "source": "hr_policy",
        "type": "factual",
    },
    {
        "id": "H05",
        "question": "Is an employee on a PIP eligible for leave encashment on separation?",
        "expected_answer": "No. Employees under a PIP at the time of separation are not eligible for leave encashment.",
        "source": "leave_policy",  # cross-document — tests retrieval across docs
        "type": "edge_case",
    },

    # ── SECURITY POLICY ───────────────────────────────────────────
    {
        "id": "S01",
        "question": "What is the minimum password length required for standard user accounts?",
        "expected_answer": "At least 12 characters, including uppercase, lowercase, numbers, and special characters.",
        "source": "security_policy",
        "type": "factual",
    },
    {
        "id": "S02",
        "question": "How often must passwords be changed for privileged accounts?",
        "expected_answer": "Every 60 days (standard accounts require changes every 90 days).",
        "source": "security_policy",
        "type": "factual",
    },
    {
        "id": "S03",
        "question": "What should an employee do if they lose their company laptop?",
        "expected_answer": "Report it to the IT security team and HR within 2 hours of discovery.",
        "source": "security_policy",
        "type": "factual",
    },
    {
        "id": "S04",
        "question": "Can employees use personal USB drives to carry company data?",
        "expected_answer": "No, unless the drive is encrypted and explicitly approved by the IT security team.",
        "source": "security_policy",
        "type": "edge_case",
    },
    {
        "id": "S05",
        "question": "What encryption standard is required for data at rest in cloud environments?",
        "expected_answer": "AES-256 encryption.",
        "source": "security_policy",
        "type": "factual",
    },

    # ── COMPANY POLICY ────────────────────────────────────────────
    {
        "id": "C01",
        "question": "What is the maximum gift value an employee can accept from an external party?",
        "expected_answer": "INR 2,000. Gifts above this must be declared to the compliance team within 5 working days.",
        "source": "company_policy",
        "type": "factual",
    },
    {
        "id": "C02",
        "question": "How long must financial records be retained?",
        "expected_answer": "A minimum of 7 years as required by law.",
        "source": "company_policy",
        "type": "factual",
    },
    {
        "id": "C03",
        "question": "Who is authorized to speak to the media on behalf of Apex Solutions?",
        "expected_answer": "Only authorized spokespersons — typically the CEO, designated executives, or the communications team.",
        "source": "company_policy",
        "type": "factual",
    },
    {
        "id": "C04",
        "question": "Can an employee accept a cash gift card worth INR 1,500 from a vendor?",
        "expected_answer": "No. Cash equivalents like gift cards must never be accepted regardless of value.",
        "source": "company_policy",
        "type": "edge_case",
    },

    # ── OUT OF SCOPE ───────────────────────────────────────────────
    {
        "id": "O01",
        "question": "What is the company's stock price today?",
        "expected_answer": "This information is not covered in the available policy documents.",
        "source": "none",
        "type": "out_of_scope",
    },
    {
        "id": "O02",
        "question": "How do I reset my laptop if it's not booting?",
        "expected_answer": "This information is not covered in the available policy documents.",
        "source": "none",
        "type": "out_of_scope",
    },
]


# ── Simple runner ──────────────────────────────────────────────────────────────

def run_eval(rag_pipeline_fn):
    """
    Pass your RAG pipeline as a callable: fn(question: str) -> str
    Prints results for manual review.
    """
    results = []
    for item in evaluationSet:
        answer = rag_pipeline_fn(item["question"])
        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected": item["expected_answer"],
            "got": answer,
        })
        print(f"[{item['id']}] {item['type'].upper()}")
        print(f"  Q: {item['question']}")
        print(f"  Expected : {item['expected_answer']}")
        print(f"  Got      : {answer}")
        print()
    return results


if __name__ == "__main__":
    # Quick sanity check — just prints the eval set
    print(f"Total eval questions: {len(evaluationSet)}")
    for item in evalationSet:
        print(f"  [{item['id']}] ({item['type']}) {item['question']}")