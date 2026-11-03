"""Multiply the adversarial seeds into variants — offline, deterministic.

Reads adversarial_seeds.jsonl, applies every mutation in mutations.py
to every llm_judge-scored attack's output, and writes
adversarial_variants.jsonl: one {"meta": ...} header line (carrying
the seeds file's sha256, so a stale variants file is detectable), then
the mutated attack cases, then the benign controls verbatim (controls
are not mutated — the siblings' convention: mutations multiply
attacks; the benign corpus stays hand-written so its FRR is readable).

Deterministic-scorer attacks (contains/regex) are not mutated either:
their payloads game string matching, and mutating them would measure
the mutation, not the scorer. They are graded in the seed suite.

Regeneration is byte-identical: fixed key order, fixed separators,
cases sorted by id. Run:
    python evals/generate_variants.py
then diff adversarial_variants.jsonl — any change is a real change.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutations import MUTATIONS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, "adversarial_seeds.jsonl")
OUT = os.path.join(HERE, "adversarial_variants.jsonl")


def load_seeds(path=SEEDS):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def dump(case):
    return json.dumps(case, ensure_ascii=False, sort_keys=True)


def build_variants(cases):
    variants = []
    for case in cases:
        if case["kind"] != "attack" or case["scorer"] != "llm_judge":
            continue
        for name, fn in sorted(MUTATIONS.items()):
            v = dict(case)
            v["id"] = f"{case['id']}__{name}"
            v["output"] = fn(case["output"])
            v["mutation"] = name
            v["seed_id"] = case["id"]
            variants.append(v)
    controls = [dict(c) for c in cases if c["kind"] == "control"]
    return sorted(variants, key=lambda c: c["id"]) + sorted(
        controls, key=lambda c: c["id"]
    )


def main():
    with open(SEEDS, "rb") as f:
        seeds_sha = hashlib.sha256(f.read()).hexdigest()
    cases = load_seeds()
    rows = build_variants(cases)
    meta = {
        "meta": {
            "generated_from": "adversarial_seeds.jsonl",
            "seeds_sha256": seeds_sha,
            "mutations": sorted(MUTATIONS),
            "rows": len(rows),
        }
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(dump(meta) + "\n")
        for row in rows:
            f.write(dump(row) + "\n")
    print(f"wrote {OUT}: {len(rows)} rows (seeds sha256 {seeds_sha[:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
