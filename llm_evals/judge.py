"""LLM-as-judge: rubric prompt, a mock stand-in, and a human-agreement check.

The rubric prompt is the actual product here — it gets used with a real
model via the `# SWAP:` seam. The mock judge is deliberately coarse
(keyword overlap, no semantics); the agreement dataset measures exactly
how far off it is, which is the point of the check.

CLI:
    python -m llm_evals.judge datasets/judge_agreement.jsonl
"""

import argparse
import json
import os
import re
import sys

# one rubric, three criteria. keep it small: judges get worse, not better,
# with ten vague criteria.
DEFAULT_RUBRIC = {
    "correctness": (
        "every checkable fact in OUTPUT matches REFERENCE. "
        "5 = all facts match; 3 = partly right or hedged; "
        "1 = contradicts REFERENCE or invents facts."
    ),
    "completeness": (
        "OUTPUT covers everything REFERENCE requires. "
        "5 = nothing missing; 3 = main point only; "
        "1 = empty, refusal, or off-topic."
    ),
    "clarity": (
        "OUTPUT is readable and in the right form for the task. "
        "5 = clear and well-formed; 3 = understandable but clumsy; "
        "1 = incoherent."
    ),
}


class JudgeConfigError(Exception):
    pass


def render_prompt(case_input, reference, output, rubric=None):
    """Build the rubric prompt. The verdict contract is fixed JSON."""
    rubric = rubric or DEFAULT_RUBRIC
    lines = [
        "You are a strict, literal evaluation judge. Grade the candidate OUTPUT",
        "against the REFERENCE using ONLY the rubric below — not vibes, not eloquence.",
        "",
        "RUBRIC — score each criterion 1 (worst) to 5 (best):",
    ]
    for name, desc in rubric.items():
        lines.append(f"- {name}: {desc}")
    lines += [
        "",
        "REFERENCE:",
        str(reference),
        "",
        "OUTPUT:",
        str(output),
        "",
        "Respond with exactly one JSON object and nothing else:",
        '{"scores": {"' + '": <1-5>, "'.join(rubric) + '": <1-5>}, "rationale": "<one sentence>"}',
        "",
        "CASE (machine-readable, same content as above):",
        "```json",
        json.dumps({"input": case_input, "reference": reference, "output": output}),
        "```",
    ]
    return "\n".join(lines)


def parse_verdict(text, rubric=None):
    """Pull the JSON verdict out of judge output. Strict — a judge that
    can't follow the format is a failed case, not a silent pass."""
    rubric = rubric or DEFAULT_RUBRIC
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:120]!r}")
    try:
        verdict = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"judge output is not valid JSON: {e}")
    if not isinstance(verdict, dict) or not isinstance(verdict.get("scores"), dict):
        raise ValueError("verdict needs a 'scores' object")
    if not isinstance(verdict.get("rationale"), str):
        raise ValueError("verdict needs a string 'rationale'")
    unknown = set(verdict["scores"]) - set(rubric)
    if unknown:
        raise ValueError(f"verdict scores unknown criteria: {sorted(unknown)}")
    missing = set(rubric) - set(verdict["scores"])
    if missing:
        raise ValueError(f"verdict missing criteria: {sorted(missing)}")
    for name, value in verdict["scores"].items():
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ValueError(f"score for {name!r} must be an int 1-5, got {value!r}")
    return {"scores": dict(verdict["scores"]), "rationale": verdict["rationale"]}


_TOKEN = re.compile(r"[a-z0-9]+")


def _band(x):
    return 5 if x >= 0.9 else 4 if x >= 0.7 else 3 if x >= 0.5 else 2 if x >= 0.3 else 1


def mock_judge_fn(prompt):
    """Deterministic stand-in for the rubric prompt. Reads the machine-readable
    CASE block like any judge would, then grades on keyword overlap — coarse
    on purpose. Paraphrase is its known blind spot (see the agreement check)."""
    m = re.search(r"```json\n(.*?)\n```", prompt, re.S)
    if not m:
        raise ValueError("mock judge: no CASE block in prompt")
    case = json.loads(m.group(1))
    ref_toks = set(_TOKEN.findall(str(case["reference"]).lower()))
    out_toks = set(_TOKEN.findall(str(case["output"]).lower()))
    overlap = ref_toks & out_toks
    precision = len(overlap) / len(out_toks) if out_toks else 0.0
    recall = len(overlap) / len(ref_toks) if ref_toks else 0.0
    # the one contradiction a keyword judge CAN catch: digits in the output
    # that never appear in the reference (wrong amounts, wrong years)
    digits_out = {t for t in out_toks if t.isdigit()}
    digits_ref = {t for t in ref_toks if t.isdigit()}
    contradiction = bool(digits_out - digits_ref)

    correctness = _band(precision)
    if contradiction:
        correctness = 1
    completeness = _band(recall)
    # terse-but-right: gets the fact, drops the context
    if len(out_toks) < 0.5 * len(ref_toks) and len(ref_toks) > 3:
        completeness = max(1, completeness - 1)
    clarity = 3 if len(str(case["output"]).split()) > 60 else 5

    scores = {"correctness": correctness, "completeness": completeness, "clarity": clarity}
    rationale = (
        f"keyword overlap p={precision:.2f} r={recall:.2f}"
        + (", digit contradiction" if contradiction else "")
    )
    return json.dumps({"scores": scores, "rationale": rationale})


def gemini_judge_fn(api_key=None, model=None):
    """# SWAP: a real model behind the same rubric prompt.

    Reads LLM_API_KEY / LLM_MODEL from the environment (same convention as
    the Render services). Raises JudgeConfigError without a key — the mock
    stays the offline default.
    """
    import urllib.request

    key = api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise JudgeConfigError("LLM_API_KEY not set — mock judge stays the default")
    model = model or os.environ.get("LLM_MODEL", "gemini-2.0-flash")

    def call(prompt):
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        return data["candidates"][0]["content"]["parts"][0]["text"]

    return call


class Judge:
    """Grades (input, reference, output) against the rubric.

    grade() returns the raw verdict + normalized average; score() applies
    the pass/fail threshold for the runner (thresholded 1.0/0.0, detail
    carries the breakdown).
    """

    def __init__(self, judge_fn=None, rubric=None, threshold=0.8):
        self.judge_fn = judge_fn or mock_judge_fn
        self.rubric = rubric or DEFAULT_RUBRIC
        self.threshold = threshold

    def grade(self, case_input, reference, output):
        prompt = render_prompt(case_input, reference, output, self.rubric)
        verdict = parse_verdict(self.judge_fn(prompt), self.rubric)
        avg = sum(verdict["scores"].values()) / (5 * len(self.rubric))
        return {**verdict, "avg": round(avg, 3)}, avg

    def score(self, case_input, reference, output):
        verdict, avg = self.grade(case_input, reference, output)
        passed = 1.0 if avg >= self.threshold else 0.0
        breakdown = ", ".join(f"{k} {v}" for k, v in verdict["scores"].items())
        detail = (
            f"{breakdown} · avg {verdict['avg']:.2f} "
            f"{'≥' if passed else '<'} {self.threshold} → "
            f"{'pass' if passed else 'fail'} — {verdict['rationale']}"
        )
        return passed, detail


AGREEMENT_FIELDS = ("id", "input", "expected", "output", "human")


def load_agreement_dataset(path, rubric=None):
    """Labelled set for the agreement check. `human.scores` are the author's
    1-5 labels per criterion, written from the spec — stand-in labels for the
    harness plumbing; a real human study replaces them, not the code."""
    rubric = rubric or DEFAULT_RUBRIC
    cases = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})")
            missing = [k for k in AGREEMENT_FIELDS if k not in case]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields {missing}")
            human = case["human"]
            if set(human.get("scores", {})) != set(rubric):
                raise ValueError(f"{path}:{lineno}: human.scores must cover {sorted(rubric)}")
            for name, value in human["scores"].items():
                if not isinstance(value, int) or not 1 <= value <= 5:
                    raise ValueError(f"{path}:{lineno}: human score {name!r} must be int 1-5")
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def measure_agreement(path, judge_fn=None, rubric=None, threshold=0.8):
    """Grade the labelled set with a judge_fn, compare against human labels.

    Returns per-case rows + aggregates: exact agreement, within-1 agreement,
    mean absolute error (overall + per criterion), and pass/fail decision
    agreement (judge avg ≥ threshold vs human mean ≥ 4.0).
    """
    rubric = rubric or DEFAULT_RUBRIC
    cases = load_agreement_dataset(path, rubric)
    judge = Judge(judge_fn=judge_fn, rubric=rubric, threshold=threshold)
    rows, errs, exact, within1 = [], [], 0, 0
    per_crit = {name: [] for name in rubric}
    decisions = 0
    for case in cases:
        verdict, avg = judge.grade(case["input"], case["expected"], case["output"])
        human = case["human"]["scores"]
        row = {"id": case["id"], "judge": verdict["scores"], "human": human}
        for name in rubric:
            err = abs(verdict["scores"][name] - human[name])
            errs.append(err)
            per_crit[name].append(err)
            row[f"err_{name}"] = err
            if err == 0:
                exact += 1
            if err <= 1:
                within1 += 1
        judge_pass = avg >= threshold
        human_pass = sum(human.values()) / len(human) >= 4.0
        row["judge_pass"] = judge_pass
        row["human_pass"] = human_pass
        if judge_pass == human_pass:
            decisions += 1
        rows.append(row)
    n = len(cases) * len(rubric)
    summary = {
        "cases": len(cases),
        "exact_agreement": round(exact / n, 3),
        "within1_agreement": round(within1 / n, 3),
        "mae": round(sum(errs) / n, 3),
        "mae_per_criterion": {k: round(sum(v) / len(v), 3) for k, v in per_crit.items()},
        "decision_agreement": round(decisions / len(cases), 3),
    }
    return rows, summary


def print_agreement(rows, summary):
    print(f"\njudge-vs-human agreement ({summary['cases']} labelled cases)")
    print(f"{'case':<12}{'criterion':<13}{'judge':>6}{'human':>6}{'err':>5}")
    for r in rows:
        first = True
        for name in ("correctness", "completeness", "clarity"):
            tag = r["id"] if first else ""
            print(f"{tag:<12}{name:<13}{r['judge'][name]:>6}{r['human'][name]:>6}{r['err_'+name]:>5}")
            first = False
        jp, hp = r["judge_pass"], r["human_pass"]
        print(f"{'':<12}{'decision':<13}{str(jp):>6}{str(hp):>6}{'✓' if jp == hp else '✗':>5}")
    print()
    print(f"exact agreement:      {summary['exact_agreement']:.3f}")
    print(f"within-1 agreement:   {summary['within1_agreement']:.3f}")
    print(f"mean absolute error:  {summary['mae']:.3f}")
    for k, v in summary["mae_per_criterion"].items():
        print(f"  mae {k:<12}: {v:.3f}")
    print(f"decision agreement:   {summary['decision_agreement']:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="judge-vs-human agreement check")
    parser.add_argument("dataset", help="path to a labelled agreement JSONL")
    parser.add_argument("--judge", default="mock", choices=("mock", "gemini"),
                        help="which judge_fn grades the set")
    args = parser.parse_args(argv)
    judge_fn = gemini_judge_fn() if args.judge == "gemini" else mock_judge_fn
    rows, summary = measure_agreement(args.dataset, judge_fn=judge_fn)
    print_agreement(rows, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
