"""Regression mode: compare two saved runs, fail CI on score drops.

Every runner invocation writes a snapshot (per-case scores + aggregates)
under results/<dataset>/<timestamp>/. This module diffs a baseline run
against a candidate run and reports what moved: overall accuracy, per-scorer
accuracy, and the per-case list of newly-failed / fixed / changed cases.

A drop in overall accuracy beyond the allowed threshold is a regression:
the CLI exits 1 so CI gates on it. New or removed case ids are reported
but never count toward the verdict — you can't regress on a case that
didn't exist before.

CLI:
    python -m llm_evals.regression results/sample/20261021-080000 results/sample/20261024-081500
    python -m llm_evals.regression baseline latest --threshold 0.02
"""

import argparse
import json
import os
import sys

CASES_FILE = "cases.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "regression.json"


def load_run(run_dir):
    """Read a saved run dir into (cases_by_id, summary)."""
    cases_path = os.path.join(run_dir, CASES_FILE)
    summary_path = os.path.join(run_dir, SUMMARY_FILE)
    if not os.path.isfile(cases_path):
        raise ValueError(f"{run_dir}: no {CASES_FILE} — not a saved run")
    cases = {}
    with open(cases_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                case = json.loads(line)
                cases[case["id"]] = case
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    return cases, summary


def _acc(cases):
    # accuracy straight from case records — summary may be stale
    if not cases:
        return None
    return round(sum(1 for c in cases.values() if c.get("passed")) / len(cases), 3)


def _by_key(cases, key):
    groups = {}
    for c in cases.values():
        groups.setdefault(c.get(key, "unknown"), []).append(c)
    out = {}
    for k, group in groups.items():
        n = len(group)
        out[k] = {"n": n, "passed": sum(1 for c in group if c.get("passed")),
                  "acc": round(sum(1 for c in group if c.get("passed")) / n, 3)}
    return out


def compare_runs(baseline_dir, candidate_dir, threshold=0.0):
    """Diff two saved runs. Returns a report dict with a 'regressed' verdict."""
    base_cases, base_summary = load_run(baseline_dir)
    cand_cases, _ = load_run(candidate_dir)

    base_ids, cand_ids = set(base_cases), set(cand_cases)
    common = sorted(base_ids & cand_ids)

    newly_failed, fixed, changed_output = [], [], []
    for cid in common:
        b, c = base_cases[cid], cand_cases[cid]
        # a case with an error field scored 0 — the regression is the drop,
        # whatever caused it; detail lives in the run dir
        if b.get("passed") and not c.get("passed"):
            newly_failed.append(cid)
        elif not b.get("passed") and c.get("passed"):
            fixed.append(cid)
        elif b.get("passed") and c.get("passed") and b.get("output") != c.get("output"):
            changed_output.append(cid)

    base_acc, cand_acc = _acc(base_cases), _acc(cand_cases)
    delta = round(cand_acc - base_acc, 3) if base_acc is not None and cand_acc is not None else None

    base_by_scorer, cand_by_scorer = _by_key(base_cases, "scorer"), _by_key(cand_cases, "scorer")
    scorer_delta = {}
    for scorer in sorted(set(base_by_scorer) | set(cand_by_scorer)):
        b = base_by_scorer.get(scorer, {}).get("acc")
        c = cand_by_scorer.get(scorer, {}).get("acc")
        scorer_delta[scorer] = {
            "baseline": b,
            "candidate": c,
            "delta": round(c - b, 3) if b is not None and c is not None else None,
        }

    regressed = delta is not None and delta < -threshold
    return {
        "baseline": os.path.basename(os.path.normpath(baseline_dir)),
        "candidate": os.path.basename(os.path.normpath(candidate_dir)),
        "threshold": threshold,
        "cases": {"baseline_n": len(base_cases), "candidate_n": len(cand_cases),
                  "compared_n": len(common),
                  "new_ids": sorted(cand_ids - base_ids),
                  "removed_ids": sorted(base_ids - cand_ids)},
        "overall": {"baseline": base_acc, "candidate": cand_acc, "delta": delta},
        "by_scorer": scorer_delta,
        "newly_failed": newly_failed,
        "fixed": fixed,
        "changed_output": changed_output,
        "regressed": regressed,
    }


def _fmt_delta(delta):
    if delta is None:
        return "n/a"
    return f"{'+' if delta > 0 else ''}{delta:.3f}"


def format_report(report):
    lines = [f"regression: {report['baseline']} -> {report['candidate']}"]
    o = report["overall"]
    lines.append(f"overall: {o['baseline']:.3f} -> {o['candidate']:.3f} "
                 f"(delta {_fmt_delta(o['delta'])}, threshold {report['threshold']})")
    for scorer, s in report["by_scorer"].items():
        if s["delta"] is not None:
            lines.append(f"  {scorer}: {s['baseline']:.3f} -> {s['candidate']:.3f} "
                         f"(delta {_fmt_delta(s['delta'])})")
    c = report["cases"]
    if c["new_ids"]:
        lines.append(f"new cases ({len(c['new_ids'])}): {', '.join(c['new_ids'][:10])}")
    if c["removed_ids"]:
        lines.append(f"removed cases ({len(c['removed_ids'])}): {', '.join(c['removed_ids'][:10])}")
    if report["newly_failed"]:
        lines.append(f"newly failed ({len(report['newly_failed'])}): "
                     f"{', '.join(report['newly_failed'][:20])}")
    if report["fixed"]:
        lines.append(f"fixed ({len(report['fixed'])}): {', '.join(report['fixed'][:20])}")
    if report["changed_output"]:
        lines.append(f"output changed but still passing ({len(report['changed_output'])}): "
                     f"{', '.join(report['changed_output'][:20])}")
    lines.append("REGRESSION" if report["regressed"] else "no regression")
    return "\n".join(lines)


def save_report(report, candidate_dir):
    path = os.path.join(candidate_dir, REPORT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="compare two saved runs; exit 1 on regression")
    parser.add_argument("baseline", help="baseline run dir (results/<dataset>/<timestamp>)")
    parser.add_argument("candidate", help="candidate run dir")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="allowed accuracy drop before failing (default 0.0 = any drop fails)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't write regression.json into the candidate run dir")
    args = parser.parse_args(argv)

    if args.threshold < 0:
        parser.error("--threshold must be >= 0")
    try:
        report = compare_runs(args.baseline, args.candidate, args.threshold)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not args.no_save:
        save_report(report, args.candidate)
    print(format_report(report))
    return 1 if report["regressed"] else 0


if __name__ == "__main__":
    sys.exit(main())
