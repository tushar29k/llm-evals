"""Offline self-test for the day-2 runner + scorers.

Plain asserts, no pytest, no network. Run from the repo root:
    python tests/run_tests.py
Exit 0 when everything passes, 1 otherwise.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_evals.mock import mock_model_fn
from llm_evals.runner import load_dataset, run_dataset, save_results
from llm_evals.scorers import score_case

FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def case(scorer, expected, params=None, cid="t-1"):
    c = {"id": cid, "input": "x", "expected": expected, "scorer": scorer}
    if params:
        c["scorer_params"] = params
    return c


def test_scorers():
    check("exact hit", score_case(case("exact", "Paris"), "Paris") == (1.0, "exact match"))
    check("exact trims whitespace", score_case(case("exact", "Paris"), "  Paris\n")[0] == 1.0)
    check("exact miss", score_case(case("exact", "Paris"), "paris")[0] == 0.0)
    check("contains hit", score_case(case("contains", "30 days"), "window is 30 days.")[0] == 1.0)
    check("contains miss", score_case(case("contains", "30 days"), "window unknown")[0] == 0.0)
    check(
        "contains any-of-list",
        score_case(case("contains", ["a", "b"]), "has b here")[0] == 1.0,
    )
    check(
        "contains case_insensitive",
        score_case(case("contains", "yes", {"case_insensitive": True}), "Yes, sure")[0] == 1.0,
    )
    check(
        "regex hit",
        score_case(case("regex", None, {"pattern": r"^ORD-\d{5}$"}), "ORD-77881")[0] == 1.0,
    )
    check(
        "regex miss",
        score_case(case("regex", None, {"pattern": r"^ORD-\d{5}$"}), "ord-77881")[0] == 0.0,
    )
    try:
        score_case(case("regex", None), "x")
        check("regex needs pattern", False, "no ValueError raised")
    except ValueError:
        check("regex needs pattern", True)
    schema = {"type": "object", "properties": {"points": {"type": "integer"}}, "required": ["points"]}
    check(
        "json_schema hit",
        score_case(case("json_schema", None, {"schema": schema}), '{"points": 1250}')[0] == 1.0,
    )
    check(
        "json_schema bad type",
        score_case(case("json_schema", None, {"schema": schema}), '{"points": "lots"}')[0] == 0.0,
    )
    check(
        "json_schema not json",
        score_case(case("json_schema", None, {"schema": schema}), "1,250 points")[0] == 0.0,
    )


def test_loader(tmp):
    bad = os.path.join(tmp, "bad.jsonl")
    with open(bad, "w") as f:
        f.write('{"id": "a-1", "input": "x", "expected": "y", "scorer": "nope"}\n')
    try:
        load_dataset(bad)
        check("loader rejects unknown scorer", False, "no ValueError")
    except ValueError:
        check("loader rejects unknown scorer", True)

    with open(bad, "w") as f:
        f.write('{"id": "a-1", "input": "x"}\n')
    try:
        load_dataset(bad)
        check("loader rejects missing fields", False, "no ValueError")
    except ValueError:
        check("loader rejects missing fields", True)

    with open(bad, "w") as f:
        f.write('{"id": "a-1", "input": "x", "expected": "y", "scorer": "exact"}\n')
        f.write('{"id": "a-1", "input": "z", "expected": "y", "scorer": "exact"}\n')
    try:
        load_dataset(bad)
        check("loader rejects duplicate ids", False, "no ValueError")
    except ValueError:
        check("loader rejects duplicate ids", True)

    with open(bad, "w") as f:
        f.write("this is not json\n")
    try:
        load_dataset(bad)
        check("loader rejects bad json", False, "no ValueError")
    except ValueError:
        check("loader rejects bad json", True)


def test_runner_end_to_end(tmp):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases = load_dataset(os.path.join(root, "datasets", "sample.jsonl"))
    check("sample dataset loads 10 cases", len(cases) == 10, f"got {len(cases)}")

    results, summary = run_dataset(cases, mock_model_fn)
    check("all cases ran", len(results) == 10)
    # the three traps fail by design: injection (005), distractor year (007), prose-not-json (008)
    check(
        "7 pass / 3 fail",
        summary["passed"] == 7 and summary["failed_ids"] == ["sample-005", "sample-007", "sample-008"],
        str(summary["failed_ids"]),
    )
    check("accuracy 0.7", summary["accuracy"] == 0.7, str(summary["accuracy"]))
    check("by_scorer present", set(summary["by_scorer"]) == {"contains", "exact", "regex", "json_schema"})
    check("every result has latency", all("latency_s" in r for r in results))

    dest = save_results(results, summary, "sample", outdir=tmp)
    lines = open(os.path.join(dest, "cases.jsonl")).read().strip().split("\n")
    check("per-case jsonl written", len(lines) == 10)
    first = json.loads(lines[0])
    check("case jsonl has output+score", "output" in first and "score" in first)
    check("summary.json written", os.path.exists(os.path.join(dest, "summary.json")))


def test_broken_model_fn(tmp):
    def boom(_):
        raise RuntimeError("model exploded")

    c = case("contains", "x", cid="boom-1")
    results, summary = run_dataset([c], boom)
    r = results[0]
    check("crash recorded not raised", r["score"] == 0.0 and r["error"] == "RuntimeError")
    check("crash detail kept", "model exploded" in r["detail"])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_scorers()
        test_loader(tmp)
        test_runner_end_to_end(tmp)
        test_broken_model_fn(tmp)
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
