"""Worked examples: point the harness at rag-service and agent-service.

Both services keep their own native eval files (rag-service's
evals/golden.jsonl, agent-service's evals/tasks.yaml). This module is the
bridge, in both directions:

1. Mock model_fns that mirror the two services' seeded behaviour, so the
   example datasets below run green offline with no service to deploy.
2. Converters that translate the native files into llm-evals cases, so a
   real golden set can be scored here without hand-rewriting it.

The converters never fabricate outputs. Tasks a single-turn model_fn
can't reproduce — agent approval gates and injection refusals — are
reported as skipped, not silently rewritten. Those outcomes are scored
in the example datasets via fixed_output instead (day-4 pattern): the
recorded answer is the thing under test, not a mock's guess at it.

Real-service swap: replace mock_rag_service_fn with a function that calls
rag-service's pipeline, and mock_agent_service_fn with one that drives
agent-service's ReAct loop — the case format doesn't change.
# SWAP: model_fn(input) -> output text, same contract as runner.py
"""

import json

# rag-service's seeded docs, condensed to the answer each golden question
# checks for — the mock answers from the same facts, not from retrieval.
# order matters: specific phrases first, generic fallbacks last.
_RAG_ANSWERS = (
    ("maternity", "Women employees are entitled to 26 weeks of paid maternity leave."),
    ("paternity", "Paternity leave is 15 days of paid leave, to be availed within 6 months."),
    ("give hr", "Employees must notify HR at least 8 weeks before the expected date of leave."),
    ("not cover", "The warranty does not cover accidental damage, water damage, or unauthorised repairs."),
    ("claim warranty", "To claim warranty service, register your product within 30 days of purchase."),
    ("warranty", "All electronics carry a 2 year manufacturer warranty."),
    ("express shipping", "Express shipping costs $12.99."),
    ("free standard shipping", "Standard shipping is free on orders over $50."),
    ("shipping", "Standard shipping within the United States takes 3 to 5 business days."),
    ("discount code", "New customers use code WELCOME10."),
    ("students", "Students get a 15 percent discount."),
    ("discount", "New customers get a 10 percent discount."),
    ("include tax", "Indian prices include GST."),
    ("clearance", "Items purchased during clearance sales are final sale and cannot be refunded."),
    ("process a refund", "Refunds are processed within 5 to 7 business days."),
    ("request a refund", "To request a refund, email support@example.com with your order ID."),
    ("refund", "Customers in India are eligible for a full refund within 30 days of purchase."),
)


def mock_rag_service_fn(case_input):
    """Keyword mock of rag-service's seeded policy KB. Unknown -> abstain."""
    text = case_input if isinstance(case_input, str) else str(case_input.get("question", ""))
    low = text.lower()
    for phrase, answer in _RAG_ANSWERS:
        if phrase in low:
            return answer
    return "I don't know."


_AGENT_ANSWERS = (
    ("months", "2 years is 24 months."),
    ("fibonacci", "The 20th Fibonacci number is 6765."),
    ("factorial", "The factorial of 6 is 720."),
    ("minutes", "1 business day is 480 minutes."),
    ("favourite colour", "I couldn't find that in the docs."),
    ("refund", "Our refund window is 30 days from purchase."),
)


def mock_agent_service_fn(case_input):
    """Keyword mock of agent-service's task set. No tools, no gates here —
    the gate tasks in the example dataset use fixed_output instead."""
    text = case_input if isinstance(case_input, str) else str(case_input)
    low = text.lower()
    for phrase, answer in _AGENT_ANSWERS:
        if phrase in low:
            return answer
    return "I couldn't find that in the docs."


def translate_rag_golden(path):
    """rag-service evals/golden.jsonl -> llm-evals cases, one per line.

    Native shape: {"question", "must_contain": [...], "expected_doc"}.
    The must_contain list becomes the case's expected list (contains
    passes on any one of them), so the check means exactly what it
    meant in rag-service.
    """
    cases = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append({
                "id": f"ragsvc-{i:03d}",
                "input": row["question"],
                "expected": row["must_contain"],
                "scorer": "contains",
                "tags": [row.get("expected_doc") or "unknown"],
                "difficulty": "easy",
                "notes": "converted from rag-service evals/golden.jsonl",
            })
    return cases


def translate_agent_tasks(tasks):
    """agent-service evals/tasks.yaml (parsed list) -> (cases, skipped).

    A task converts when its check is a final-answer substring. Tasks
    with expect_approval or expect_refusal are skipped — their verdict
    lives in the gate/audit log, not in answer text, and a single-turn
    model_fn can't reproduce them. Skipped ids are returned, never
    dropped silently.
    """
    cases, skipped = [], []
    for task in tasks:
        tid = task["id"]
        if task.get("expect_approval") or task.get("expect_refusal"):
            skipped.append(tid)
            continue
        cases.append({
            "id": f"agentsvc-{tid}",
            "input": task["question"],
            "expected": task["expected_contains"],
            "scorer": "contains",
            "tags": ["agent-task"],
            "difficulty": "easy",
            "notes": "converted from agent-service evals/tasks.yaml",
        })
    return cases, skipped


def load_agent_tasks_yaml(path):
    """Parse agent-service's tasks.yaml. Needs pyyaml (agent-service dep)."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - env gap, not code gap
        raise RuntimeError("pyyaml is required to read agent-service tasks.yaml") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["tasks"]
