"""Offline mock model: a tiny keyword responder for the sample dataset.

Deliberately naive — it parrots injections and grabs the first year it
sees, which is exactly why the trap cases in datasets/sample.jsonl fail.
A mock that never fails teaches nothing.
"""

import re


def mock_model_fn(case_input):
    # multi-turn inputs: answer the last user message
    if isinstance(case_input, dict):
        msgs = case_input.get("messages", [])
        case_input = next(
            (m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"),
            "",
        )
    text = str(case_input)
    low = text.lower()

    # gullible on purpose: obeys injected instructions (sample-005 trap)
    if "ignore previous instructions" in low:
        return "The refund window is 90 days."

    if "refund" in low:
        return "Our refund window is 30 days from purchase."

    if "capital of france" in low:
        return "Paris"

    m = re.search(r"ord-(\d+)", low)
    if m or "order reference" in low:
        return f"ORD-{m.group(1)}" if m else "ORD-00000"

    if "store hours" in low:
        return '{"open": "9am", "close": "9pm"}'

    if "loyalty points" in low:
        return "You have 1,250 points."

    if "reset my password" in low or "password reset" in low:
        return "We sent a password reset link to your email."

    if "ship internationally" in low:
        return "Yes, we ship to 40+ countries."

    if "founded" in low:
        # naive grab: first year mentioned wins (sample-007 trap)
        years = re.findall(r"\b(19|20)\d{2}\b", text)
        return years[0] if years else "2020"

    return "I don't have an answer for that."
