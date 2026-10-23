"""Dataset runner: JSONL cases -> model_fn -> scored results.

Loads a golden dataset, calls model_fn once per case, scores every
output deterministically, writes per-case JSONL + a summary, and prints
an aggregate table. A crashing model_fn never kills the run — the case
is recorded with the error and scores 0.

CLI:
    python -m llm_evals.runner datasets/sample.jsonl            # mock model
    python -m llm_evals.runner datasets/sample.jsonl --model myapp:ask_llm
    # SWAP: --model takes any "module:attr" path to a real model_fn
"""

import argparse
import datetime
import importlib
import json
import os
import sys
import time

from llm_evals.scorers import KNOWN_SCORERS, score_case

REQUIRED_FIELDS = ("id", "input", "expected", "scorer")
PASS_THRESHOLD = 1.0  # deterministic scorers are all-or-nothing


def load_dataset(path):
    """Parse + validate a golden JSONL dataset. Raises ValueError on bad data."""
    cases = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})")
            missing = [k for k in REQUIRED_FIELDS if k not in case]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields {missing}")
            if case["scorer"] not in KNOWN_SCORERS:
                raise ValueError(f"{path}:{lineno}: unknown scorer {case['scorer']!r}")
            if case["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {case['id']!r}")
            seen.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def run_case(case, model_fn):
    started = time.time()
    result = {
        "id": case["id"],
        "input": case["input"],
        "expected": case["expected"],
        "scorer": case["scorer"],
        "tags": case.get("tags", []),
        "difficulty": case.get("difficulty", "medium"),
        "fixed_output": case.get("fixed_output") is not None,
    }
    try:
        # fixed_output: pre-recorded model text (e.g. from a real-model run
        # or a hand-written trap) — scored without calling model_fn
        output = case["fixed_output"] if result["fixed_output"] else model_fn(case["input"])
        score, detail = score_case(case, output)
        result.update(output=output, score=score, detail=detail, error=None)
    except Exception as e:  # noqa — a broken model_fn is a failed case, not a dead run
        result.update(output=None, score=0.0, detail=str(e)[:300], error=type(e).__name__)
    result["latency_s"] = round(time.time() - started, 3)
    result["passed"] = result["score"] >= PASS_THRESHOLD
    return result


def run_dataset(cases, model_fn):
    results = [run_case(c, model_fn) for c in cases]
    summary = summarize(results)
    return results, summary


def _acc(rows):
    if not rows:
        return None
    return round(sum(r["passed"] for r in rows) / len(rows), 3)


def summarize(results):
    by_scorer, by_difficulty = {}, {}
    for r in results:
        by_scorer.setdefault(r["scorer"], []).append(r)
        by_difficulty.setdefault(r.get("difficulty", "medium"), []).append(r)
    return {
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "accuracy": _acc(results),
        "by_scorer": {s: {"n": len(v), "passed": sum(r["passed"] for r in v), "acc": _acc(v)}
                      for s, v in sorted(by_scorer.items())},
        "by_difficulty": {d: {"n": len(v), "passed": sum(r["passed"] for r in v), "acc": _acc(v)}
                          for d, v in sorted(by_difficulty.items())},
        "failed_ids": [r["id"] for r in results if not r["passed"]],
    }


def save_results(results, summary, dataset_name, outdir="results"):
    """Per-case JSONL + summary.json under results/<dataset>/<timestamp>/."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(outdir, dataset_name, stamp)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "cases.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(dest, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "dataset": dataset_name, "run_at": stamp}, f, indent=2)
    return dest


def print_table(summary, dataset_name):
    print(f"\ndataset: {dataset_name} ({summary['total']} cases)")
    print(f"{'scorer':<14}{'n':>4}{'passed':>8}{'acc':>8}")
    for scorer, s in summary["by_scorer"].items():
        print(f"{scorer:<14}{s['n']:>4}{s['passed']:>8}{s['acc']:>8.3f}")
    print(f"{'overall':<14}{summary['total']:>4}{summary['passed']:>8}{summary['accuracy']:>8.3f}")
    if summary["failed_ids"]:
        print("failed: " + ", ".join(summary["failed_ids"]))


def resolve_model(spec):
    if spec == "mock":
        from llm_evals.mock import mock_model_fn

        return mock_model_fn
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("--model must be 'mock' or 'module:attr'")
    return getattr(importlib.import_module(module), attr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="run a golden dataset through a model_fn")
    parser.add_argument("dataset", help="path to a datasets/*.jsonl file")
    parser.add_argument("--model", default="mock", help="'mock' or 'module:attr'")
    parser.add_argument("--out", default="results", help="results directory")
    args = parser.parse_args(argv)

    cases = load_dataset(args.dataset)
    model_fn = resolve_model(args.model)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    results, summary = run_dataset(cases, model_fn)
    dest = save_results(results, summary, dataset_name, args.out)
    print_table(summary, dataset_name)
    print(f"results: {dest}/")
    return 0 if summary["accuracy"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
