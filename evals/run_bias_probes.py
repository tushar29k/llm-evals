"""Bias probes for the judge — measured, not asserted.

The toolkit's judge is absolute (one output, one rubric), not pairwise,
so classic position bias has nothing to attach to: there is no second
candidate whose order could matter. What an absolute keyword judge
*can* have — and what these probes measure on the mock — is:

  length   the same correct answer at growing word counts, padded two
           ways: by repeating the reference sentence (unique-token set
           frozen, so only the clarity rule can move) and by appending
           neutral sentences (precision dilutes as vocabulary grows).
           The probe prints the exact word count where each effect
           bites instead of gesturing at "length bias".
  order    the reference sentence verbatim, reversed, and shuffled.
           A token-set judge cannot tell these apart; the probe shows
           the three verdicts side by side.
  wording  a ladder from verbatim to paraphrase to spelled-out digits
           to terse — the same fact at decreasing keyword overlap, so
           the mock's semantic gradient is a measured staircase.

Deterministic, report-only, exit 0. The numbers feed RESEARCH.md and
adversarial-report.md (via make_report.py).

    python evals/run_bias_probes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_evals.judge import Judge  # noqa: E402

REF = "Our refund window is 30 days from purchase."
QUESTION = "What is the refund window?"
NEUTRAL = ("The weather in the city was mild that week and the trains "
           "ran on time.")


def words(text):
    return len(str(text).split())


def grade(judge, output):
    verdict, avg = judge.grade(QUESTION, REF, output)
    return verdict["scores"], avg


def length_ladder(judge):
    rows = []
    # repetition padding: token set frozen at the reference's, only the
    # word count (and therefore the clarity rule) can move the verdict
    for reps in (1, 4, 7, 8, 12, 20):
        out = " ".join([REF] * reps)
        scores, avg = grade(judge, out)
        rows.append(("repeat", words(out), scores, avg))
    # neutral padding: each appended sentence adds fresh vocabulary, so
    # keyword precision dilutes while recall stays 1.0
    for reps in (0, 1, 3, 6, 12):
        out = REF + " " + " ".join([NEUTRAL] * reps) if reps else REF
        scores, avg = grade(judge, out)
        rows.append(("neutral", words(out), scores, avg))
    return rows


def order_probes(judge):
    toks = REF.rstrip(".").split()
    variants = [
        ("verbatim", REF),
        ("reversed", " ".join(reversed(toks)) + "."),
        ("shuffled", " ".join(toks[::2] + toks[1::2]) + "."),
    ]
    return [(name, *grade(judge, out)) for name, out in variants]


def wording_ladder(judge):
    rungs = [
        ("verbatim", REF),
        ("light paraphrase",
         "Our refund window is 30 days, counted from the purchase date."),
        ("full paraphrase",
         "You have 30 days from purchase to request a refund."),
        ("terse", "30 days"),
    ]
    return [(name, *grade(judge, out)) for name, out in rungs]


def main():
    judge = Judge()
    print("== length ladder (correct answer, growing word count) ==")
    print(f"{'padding':<9}{'words':>6}{'corr':>6}{'comp':>6}{'clar':>6}{'avg':>7}")
    for pad, n, s, avg in length_ladder(judge):
        print(f"{pad:<9}{n:>6}{s['correctness']:>6}{s['completeness']:>6}"
              f"{s['clarity']:>6}{avg:>7.3f}")
    print("\n== order probes (same tokens, different order) ==")
    for name, s, avg in order_probes(judge):
        print(f"{name:<10} correctness {s['correctness']}  "
              f"completeness {s['completeness']}  clarity {s['clarity']}  "
              f"avg {avg:.3f}")
    print("\n== wording ladder (same fact, decreasing keyword overlap) ==")
    for name, s, avg in wording_ladder(judge):
        print(f"{name:<17} correctness {s['correctness']}  "
              f"completeness {s['completeness']}  clarity {s['clarity']}  "
              f"avg {avg:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
