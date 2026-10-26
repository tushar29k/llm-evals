# Release notes

## 0.7.0 — 2026-10-26: CI + release notes (week 1 complete)

The harness now gates itself. `.github/workflows/evals.yml` runs on
every push and PR to main with zero network and zero model keys:

- `tests/run_tests.py` — 132 checks, all green offline.
- Both worked-example datasets gated: rag-service example 8/8 and
  agent-service example 8/8, each diffed against a checked-in baseline
  in `snapshots/<dataset>/` — any accuracy drop fails the build.
- Trap datasets (`sample.jsonl` 7/10, `rag_sample.jsonl` 4/7) and the
  judge-agreement check run as informational steps only: their failing
  cases are traps by design, so they report numbers without gating.
- The test suite covers the CI contract itself: snapshots load, hold
  8 passing cases each, and a fresh run diffs clean against them; the
  workflow file still contains the gated steps.

Week 1 in one line: golden datasets in, scored results out,
deterministic scorers first, LLM-as-judge second, RAG scorers for
retrieval-vs-generation diagnosis, regression snapshots that fail CI
on drops, worked examples on real projects, and now the Action that
runs all of it. What didn't change: everything still runs offline
against mocks, and real models stay behind the `# SWAP:` seams.

## How week 1 built up

- Day 1: repo scaffold, eval philosophy, golden-dataset format spec.
- Day 2: dataset runner (JSONL → `model_fn` → scored results,
  per-case JSONL + summary, exit 1 below 1.000) and deterministic
  scorers (`exact` / `contains` / `regex` / `json_schema`).
- Day 3: LLM-as-judge with rubric prompt + human-agreement check
  (exact 0.611, within-1 0.861, decision agreement 0.917 on the
  12-case labelled set — the mock's known blind spot is paraphrase).
- Day 4 (0.4.0): RAG scorers — faithfulness, answer relevance,
  context precision — plus `fixed_output` cases for recorded outcomes.
- Day 5 (0.5.0): regression mode — diff two runs, fail on drops,
  one-command gate via `runner.py --compare-with`.
- Day 6 (0.6.0): worked examples — rag-service and agent-service
  example datasets (8 cases each, 8/8 with paired mocks) and
  converters for the native eval files.
