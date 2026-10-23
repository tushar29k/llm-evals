"""Offline self-test for the harness (runner + scorers + judge + RAG scorers).

Plain asserts, no pytest, no network. Run from the repo root:
    python tests/run_tests.py
Exit 0 when everything passes, 1 otherwise.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_evals.mock import mock_model_fn, mock_rag_model_fn
from llm_evals.rag_scorers import _is_abstention, split_claims
from llm_evals.runner import load_dataset, run_dataset, save_results
from llm_evals.scorers import score_case
from llm_evals.judge import (
    Judge,
    JudgeConfigError,
    gemini_judge_fn,
    load_agreement_dataset,
    measure_agreement,
    mock_judge_fn,
    parse_verdict,
    render_prompt,
)

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


def test_judge_prompt():
    p = render_prompt("q?", "Paris", "Paris")
    check(
        "prompt names all criteria",
        all(c in p for c in ("correctness", "completeness", "clarity")),
    )
    check(
        "prompt carries reference+output",
        "REFERENCE:" in p and "OUTPUT:" in p and p.count("Paris") >= 2,
    )
    check("prompt states json contract", '"scores"' in p and '"rationale"' in p)


def test_parse_verdict():
    v = parse_verdict(
        '{"scores": {"correctness": 5, "completeness": 4, "clarity": 5}, "rationale": "ok"}'
    )
    check(
        "parse clean json",
        v["scores"]["completeness"] == 4 and v["rationale"] == "ok",
    )
    v = parse_verdict(
        'Here is my verdict:\n'
        '{"scores": {"correctness": 2, "completeness": 2, "clarity": 3}, "rationale": "meh"}\n'
        "Done."
    )
    check("parse prose-wrapped json", v["scores"]["correctness"] == 2)
    bad_cases = [
        ("no json here", "rejects no-json"),
        (
            '{"scores": {"correctness": 6, "completeness": 5, "clarity": 5}, "rationale": "x"}',
            "rejects out-of-range score",
        ),
        (
            '{"scores": {"correctness": 5, "completeness": 5}, "rationale": "x"}',
            "rejects missing criterion",
        ),
        (
            '{"scores": {"correctness": 5, "completeness": 5, "clarity": 5, "vibes": 5}, "rationale": "x"}',
            "rejects unknown criterion",
        ),
        (
            '{"scores": {"correctness": 5, "completeness": 5, "clarity": 5}}',
            "rejects missing rationale",
        ),
    ]
    for bad, name in bad_cases:
        try:
            parse_verdict(bad)
            check(name, False, "no ValueError raised")
        except ValueError:
            check(name, True)


def test_mock_judge():
    judge = Judge()
    good, avg_good = judge.grade("capital?", "Paris", "Paris")
    bad, avg_bad = judge.grade("capital?", "Paris", "Lyon")
    check("mock orders good above bad", avg_good > avg_bad, f"{avg_good} vs {avg_bad}")
    check(
        "mock verdict shape",
        set(good["scores"]) == {"correctness", "completeness", "clarity"}
        and isinstance(good["rationale"], str),
    )
    s, detail = judge.score("capital?", "Paris", "Paris")
    check("judge score passes good output", s == 1.0 and "pass" in detail, detail)
    s, detail = judge.score("capital?", "Paris", "Lyon")
    check("judge score fails wrong output", s == 0.0 and "fail" in detail, detail)


def test_llm_judge_scorer():
    c = {"id": "j-1", "input": "q", "expected": "Paris", "scorer": "llm_judge"}
    check("llm_judge passes exact", score_case(c, "Paris")[0] == 1.0)
    check("llm_judge fails wrong fact", score_case(c, "Lyon")[0] == 0.0)
    check("llm_judge detail carries breakdown", "correctness" in score_case(c, "Paris")[1])
    check(
        "llm_judge threshold param",
        score_case({**c, "scorer_params": {"threshold": 0.99}}, "Paris")[0] == 1.0,
    )
    # gemini selection without a key is a failed case, not a dead run
    env = os.environ.pop("LLM_API_KEY", None)
    try:
        s, detail = score_case({**c, "scorer_params": {"judge": "gemini"}}, "Paris")
        check("gemini judge without key fails case", s == 0.0 and "judge error" in detail, detail)
    finally:
        if env is not None:
            os.environ["LLM_API_KEY"] = env


def test_llm_judge_broken():
    import llm_evals.judge as jmod

    real = jmod.mock_judge_fn

    def boom(prompt):
        raise RuntimeError("judge exploded")

    jmod.mock_judge_fn = boom
    try:
        c = {"id": "j-9", "input": "q", "expected": "Paris", "scorer": "llm_judge"}
        s, detail = score_case(c, "Paris")
        check("broken judge is failed case", s == 0.0 and "judge error" in detail, detail)
    finally:
        jmod.mock_judge_fn = real


def test_gemini_judge_no_key():
    # offline: must refuse clearly, never hang on the network
    env = os.environ.pop("LLM_API_KEY", None)
    try:
        gemini_judge_fn()
        check("gemini judge needs a key", False, "no JudgeConfigError")
    except JudgeConfigError:
        check("gemini judge needs a key", True)
    finally:
        if env is not None:
            os.environ["LLM_API_KEY"] = env


def test_agreement(tmp):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "datasets", "judge_agreement.jsonl")
    rows, summary = measure_agreement(path)
    check("agreement loads 12 cases", summary["cases"] == 12, str(summary["cases"]))
    check("agreement rows have judge+human", all("judge" in r and "human" in r for r in rows))
    check(
        "mock within-1 agreement sane",
        summary["within1_agreement"] >= 0.8,
        str(summary["within1_agreement"]),
    )
    check("mock mae sane", summary["mae"] <= 0.6, str(summary["mae"]))
    check(
        "mock decision agreement sane",
        summary["decision_agreement"] >= 0.85,
        str(summary["decision_agreement"]),
    )

    bad = os.path.join(tmp, "agree-bad.jsonl")
    with open(bad, "w") as f:
        f.write('{"id": "x-1", "input": "q", "expected": "y", "output": "z"}\n')
    try:
        load_agreement_dataset(bad)
        check("agreement loader needs human labels", False, "no ValueError")
    except ValueError:
        check("agreement loader needs human labels", True)


def test_rag_claims():
    claims = split_claims("Our window is 30 days [1]. Refunds go back to the card.")
    check(
        "claims split + citations stripped",
        claims == ["Our window is 30 days", "Refunds go back to the card"],
        str(claims),
    )
    check("short fragments dropped", split_claims("Yes. No.") == [])
    check("abstention phrases recognised",
          _is_abstention("I don't know") and not _is_abstention("Our window is 30 days"))


RAG_CTX = ["Our refund window is 30 days from purchase. Refunds go back to the original payment method."]
RAG_Q = "What is the refund window?"
RAG_REF = "30 days from purchase, back to the original payment method."


def rag_case(scorer, question, context, cid="r-1", expected=RAG_REF, params=None, fixed=None):
    c = {"id": cid, "input": {"question": question, "context": context},
         "expected": expected, "scorer": scorer}
    if params:
        c["scorer_params"] = params
    if fixed is not None:
        c["fixed_output"] = fixed
    return c


def test_rag_faithfulness():
    c = rag_case("rag_faithfulness", RAG_Q, RAG_CTX)
    s, d = score_case(c, "Our refund window is 30 days from purchase.")
    check("faithfulness grounded passes", s == 1.0, d)
    c = rag_case("rag_faithfulness", RAG_Q, RAG_CTX, cid="r-2")
    s, d = score_case(
        c, "Our refund window is 30 days from purchase. The company was founded in 1998 by two friends.")
    check("faithfulness invented claim fails", s == 0.0 and "0.50" in d, d)
    c = rag_case("rag_faithfulness", RAG_Q, RAG_CTX, cid="r-3")
    s, d = score_case(c, "I don't know.")
    check("faithfulness abstention passes", s == 1.0, d)
    c = rag_case("rag_faithfulness", RAG_Q, RAG_CTX, cid="r-4")
    s, d = score_case(c, "Yes.")
    check("faithfulness vacuous answer passes", s == 1.0 and "no checkable claims" in d, d)
    c = rag_case("rag_faithfulness", RAG_Q, RAG_CTX, cid="r-5", params={"threshold": 0.4})
    s, d = score_case(
        c, "Our refund window is 30 days from purchase. The company was founded in 1998 by two friends.")
    check("faithfulness threshold override", s == 1.0, d)
    env = os.environ.pop("LLM_API_KEY", None)
    try:
        c = rag_case("rag_faithfulness", RAG_Q, RAG_CTX, cid="r-6", params={"judge": "gemini"})
        s, d = score_case(c, "Our refund window is 30 days from purchase.")
        check("gemini claim check without key fails case", s == 0.0 and "judge error" in d, d)
    finally:
        if env is not None:
            os.environ["LLM_API_KEY"] = env


def test_rag_answer_relevance():
    c = rag_case("rag_answer_relevance", RAG_Q, RAG_CTX)
    s, d = score_case(c, "Our refund window is 30 days from purchase.")
    check("relevance on-topic passes", s == 1.0, d)
    c = rag_case("rag_answer_relevance", RAG_Q, RAG_CTX, cid="r-7")
    s, d = score_case(c, "Loyalty points expire 12 months after they are earned.")
    check("relevance off-topic fails", s == 0.0 and "token F1 0.00" in d, d)
    c = rag_case("rag_answer_relevance", RAG_Q, RAG_CTX, cid="r-8")
    s, d = score_case(c, "")
    check("relevance empty answer fails", s == 0.0, d)


def test_rag_context_precision():
    ctx2 = RAG_CTX + ["Refunds are issued to the original payment method within 30 days of purchase.",
                      "Loyalty points expire 12 months after they are earned."]
    c = rag_case("rag_context_precision", RAG_Q, ctx2, cid="r-9")
    s, d = score_case(c, "anything")
    check("precision 2/3 passes", s == 1.0 and "2/3" in d, d)
    ctx5 = RAG_CTX + [
        "Loyalty points expire 12 months after they are earned.",
        "Our stores are open from 9am to 9pm every day.",
        "We ship to more than 40 countries worldwide.",
        "Password resets are sent to your email within minutes.",
    ]
    c = rag_case("rag_context_precision", RAG_Q, ctx5, cid="r-10")
    s, d = score_case(c, "anything")
    check("precision 1/5 fails", s == 0.0 and "1/5" in d, d)
    c = rag_case("rag_context_precision", RAG_Q, [], cid="r-11")
    s, d = score_case(c, "anything")
    check("precision no chunks fails", s == 0.0 and "no chunks" in d, d)


def test_rag_input_validation():
    try:
        score_case(case("rag_faithfulness", "y"), "out")
        check("rag scorer rejects string input", False, "no ValueError")
    except ValueError:
        check("rag scorer rejects string input", True)
    c = {"id": "r-12", "input": {"question": "q", "context": "notalist"},
         "expected": "x", "scorer": "rag_faithfulness"}
    try:
        score_case(c, "out")
        check("rag scorer rejects non-list context", False, "no ValueError")
    except ValueError:
        check("rag scorer rejects non-list context", True)


def test_mock_rag_model():
    inp = {"question": RAG_Q,
           "context": ["Loyalty points expire 12 months after they are earned.",
                       "Our refund window is 30 days from purchase. Refunds go back to the original payment method."]}
    out = mock_rag_model_fn(inp)
    check("rag mock picks the refund chunk",
          out == "Our refund window is 30 days from purchase.", out)
    out = mock_rag_model_fn({"question": RAG_Q, "context": ["Loyalty points expire."]})
    check("rag mock abstains with no overlap", out == "I don't know.", out)


def test_fixed_output():
    def boom(_):
        raise RuntimeError("model exploded")

    c = {"id": "f-1", "input": "x", "expected": "Paris", "scorer": "exact",
         "fixed_output": "Paris"}
    results, _ = run_dataset([c], boom)
    r = results[0]
    check("fixed_output bypasses model_fn",
          r["score"] == 1.0 and r["fixed_output"] is True and r["error"] is None)
    c = {"id": "f-2", "input": "x", "expected": "Paris", "scorer": "exact"}
    results, _ = run_dataset([c], mock_model_fn)
    check("fixed_output flag false by default", results[0]["fixed_output"] is False)


def test_rag_sample_end_to_end():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases = load_dataset(os.path.join(root, "datasets", "rag_sample.jsonl"))
    check("rag sample loads 7 cases", len(cases) == 7, f"got {len(cases)}")
    results, summary = run_dataset(cases, mock_rag_model_fn)
    check("all rag cases ran", len(results) == 7)
    # the traps fail by design: hallucinated claim (002), off-topic answer (004),
    # junk-heavy retrieval (006)
    check(
        "4 pass / 3 fail",
        summary["passed"] == 4
        and sorted(summary["failed_ids"]) == ["rag-002", "rag-004", "rag-006"],
        str(summary["failed_ids"]),
    )
    check("rag accuracy 0.571", summary["accuracy"] == 0.571, str(summary["accuracy"]))
    check("rag scorers in the table", set(summary["by_scorer"]) == {
        "rag_faithfulness", "rag_answer_relevance", "rag_context_precision"})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_scorers()
        test_loader(tmp)
        test_runner_end_to_end(tmp)
        test_broken_model_fn(tmp)
        test_judge_prompt()
        test_parse_verdict()
        test_mock_judge()
        test_llm_judge_scorer()
        test_llm_judge_broken()
        test_gemini_judge_no_key()
        test_agreement(tmp)
        test_rag_claims()
        test_rag_faithfulness()
        test_rag_answer_relevance()
        test_rag_context_precision()
        test_rag_input_validation()
        test_mock_rag_model()
        test_fixed_output()
        test_rag_sample_end_to_end()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
