"""Adversarial runner for the judge suites — report-only.

Grades every case in a suite file (adversarial_seeds.jsonl or
adversarial_variants.jsonl) with the toolkit's own pipeline: llm_judge
cases through llm_evals.judge.Judge with the default mock judge and
the standard 0.8 threshold, deterministic cases through
llm_evals.scorers.score_case. Nothing here patches, wraps or replaces
any toolkit code — the pipeline grades exactly as it grades in
production runs; this runner only adds ground truth (should_pass,
clean_output) and the frozen classification in scoring.py.

Report-only by design: exit code is always 0, no snapshot is gated.
The harness's CI gate stays tests/run_tests.py + the baselines.

    python evals/run_adversarial.py            # seeds + variants
    python evals/run_adversarial.py seeds      # one suite
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_evals.judge import Judge  # noqa: E402
from llm_evals.scorers import score_case  # noqa: E402
import scoring  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "eval-history.jsonl")
THRESHOLD = 0.8

SUITES = {
    "seeds": "adversarial_seeds.jsonl",
    "variants": "adversarial_variants.jsonl",
}


def load_suite(name):
    path = os.path.join(HERE, SUITES[name])
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "meta" in row:  # variants header line
                continue
            cases.append(row)
    return cases


def grade(case, judge):
    """Run one case through the pipeline. Returns the outcome dict
    scoring.classify consumes: passed, avg, clean_avg, error."""
    if case["scorer"] == "llm_judge":
        try:
            _, avg = judge.grade(case["input"], case["expected"], case["output"])
        except Exception as e:  # a judge error is data, not a dead run
            return {"passed": False, "avg": None, "clean_avg": None,
                    "error": f"{type(e).__name__}: {e}"}
        clean_avg = None
        if case.get("clean_output") is not None:
            try:
                _, clean_avg = judge.grade(
                    case["input"], case["expected"], case["clean_output"])
            except Exception:
                clean_avg = None
        return {"passed": avg >= THRESHOLD, "avg": round(avg, 3),
                "clean_avg": clean_avg, "error": None}
    try:
        score, _detail = score_case(case, case["output"])
        return {"passed": score == 1.0, "avg": None, "clean_avg": None,
                "error": None}
    except Exception as e:
        return {"passed": False, "avg": None, "clean_avg": None,
                "error": f"{type(e).__name__}: {e}"}


def run_suite(name, judge=None):
    judge = judge or Judge(threshold=THRESHOLD)
    outcomes = []
    for case in load_suite(name):
        outcome = grade(case, judge)
        outcomes.append((case, outcome, scoring.classify(case, outcome)))
    return outcomes


def print_suite(name, outcomes):
    agg = scoring.aggregate(outcomes)
    a, b = agg["attacks"], agg["benign"]
    print(f"\n== {name}: {agg['n']} runs ==")
    print(f"{'case':<28}{'class':<10}{'avg':>6}{'clean':>6}  outcome")
    for case, outcome, cls in outcomes:
        avg = f"{outcome['avg']:.2f}" if outcome.get("avg") is not None else "—"
        clean = (f"{outcome['clean_avg']:.2f}"
                 if outcome.get("clean_avg") is not None else "—")
        print(f"{case['id']:<28}{case['kind']:<10}{avg:>6}{clean:>6}  {cls}")
    print(f"attacks: n={a['n']}  ASR {scoring.fmt_rate(a['asr'], a['n'])}  "
          f"blocked {a['blocked']}  evaded {a['evaded']}  "
          f"mean inflation {a['mean_inflation'] if a['mean_inflation'] is None else round(a['mean_inflation'], 3)} (n={a['inflation_n']})")
    print(f"benign:  n={b['n']}  FRR {scoring.fmt_rate(b['frr'], b['n'])}  "
          f"completion {scoring.fmt_rate(b['completion'], b['n'])}")
    print(f"decision accuracy: {scoring.fmt_rate(agg['decision_accuracy'], agg['n'])}")
    return agg


def append_history(suite, agg):
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": f"adversarial-{suite}",
        "attacks_n": agg["attacks"]["n"],
        "asr": agg["attacks"]["asr"],
        "block_rate": agg["attacks"]["block_rate"],
        "benign_n": agg["benign"]["n"],
        "frr": agg["benign"]["frr"],
        "decision_accuracy": agg["decision_accuracy"],
        "metrics_version": scoring.METRICS_VERSION,
    }
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    names = argv or ["seeds", "variants"]
    judge = Judge(threshold=THRESHOLD)
    for name in names:
        outcomes = run_suite(name, judge)
        agg = print_suite(name, outcomes)
        append_history(name, agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
