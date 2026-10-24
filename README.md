# llm-evals

A standalone eval harness for LLM apps. One dataset format, one runner,
scorers you can trust, and a regression mode that fails CI when scores drop.

This grew out of four projects (a RAG service, a ReAct agent, an LLM
microservice, a PR review agent) that all ended up re-inventing the same
wheel: golden questions, a runner, some scoring, a snapshot to compare
against. This repo is the wheel, done once and done right.

## The eval philosophy

Most LLM evals fail in boring ways. The model changes, the prompt changes,
nobody notices quality drifted until users complain. The fix isn't fancier
metrics — it's boring infrastructure:

1. **Golden datasets are the product.** A good eval is 50 well-chosen
   cases, not a clever metric. Cases should cover the happy path, the
   known edge cases, and the traps that burned you before. Every case has
   an expected outcome written down *before* you run anything.
2. **Deterministic first, judge second.** Exact match, substring, regex,
   JSON-schema validation — these cost nothing and never argue. LLM-as-judge
   is for the cases determinism can't score (tone, helpfulness, "did it
   actually answer the question"). When you use a judge, measure the judge:
   check its agreement with human labels on a fixed set, and re-check when
   you swap the judge model.
3. **Offline by default.** Every eval in this harness runs against a mock
   model function with zero network. Real-model runs are the same code
   path with a different `model_fn` (marked `# SWAP:` where the seam is).
   If your evals need the network, you won't run them — and evals you
   don't run are decoration.
4. **Regression, not vibes.** Every run writes a snapshot (per-case scores
   + aggregates). CI compares the fresh snapshot against the checked-in
   one and fails on drops. "It felt fine" is not a test result.
5. **Scores need context.** A bare 0.87 means nothing. Snapshots record the
   model, the prompt version, the dataset version, and the date — so a
   number from March is comparable to a number from June.

## Golden-dataset format

Datasets live in `datasets/` as JSONL. One case per line, UTF-8, no
trailing commas, parseable with `json.loads` per line. Every case has
exactly these fields:

```json
{
  "id": "rag-014",
  "input": "What is the refund window?",
  "expected": "30 days",
  "scorer": "contains",
  "tags": ["faq", "happy-path"],
  "difficulty": "easy",
  "notes": "planted 2026-10-20: the classic support question"
}
```

Field reference:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique within the dataset, stable forever. Format `<dataset>-<nnn>`. Never reuse an id after deleting a case. |
| `input` | yes | What goes to the model. A string for chat/prompt tasks, or `{"messages": [...]}` for multi-turn. |
| `expected` | yes | The reference outcome. A string, a JSON object (for structured output), or a list of acceptable strings. |
| `scorer` | yes | Which scorer grades this case: `exact`, `contains`, `regex`, `json_schema`, `llm_judge`, `rag_faithfulness`, `rag_answer_relevance`, `rag_context_precision`. |
| `scorer_params` | no | Scorer-specific knobs, e.g. `{"pattern": "..."}` for regex, `{"schema": {...}}` for json_schema, `{"threshold": 0.8, "judge": "mock"}` for llm_judge and the rag scorers. |
| `tags` | no | Free-form labels for slicing results (`faq`, `adversarial`, `regression-trap`). |
| `difficulty` | no | `easy` / `medium` / `hard` — for weighting and for "did we break the easy stuff" checks. |
| `notes` | no | Why this case exists. Future you will thank present you. |
| `version` | no | Dataset schema version, defaults to `1`. Bump when the format changes. |

Rules that keep datasets honest:

- `expected` is written from the *spec*, not from model output. If you
  copy a model answer into `expected`, you've built a change detector,
  not an eval.
- Every adversarial/trap case gets a `notes` line explaining the trap.
- Datasets are versioned by git, not by filename. `golden.jsonl` is the
  file; history is the log.

## Usage

Run the bundled sample dataset against the offline mock (no network, no keys):

```
python -m llm_evals.runner datasets/sample.jsonl
```

You get an aggregate table (accuracy overall, per scorer, per difficulty),
plus per-case results under `results/sample/<timestamp>/` (`cases.jsonl`
with every output/score/detail, `summary.json` with the aggregates).
The exit code is 0 only at 1.000 accuracy — anything less is a signal,
and day 5 turns that signal into the CI gate.

Plug in a real model with `--model`:

```
python -m llm_evals.runner datasets/sample.jsonl --model myapp:ask_llm
# SWAP: any "module:attr" path whose function takes case["input"]
# (a string, or {"messages": [...]}) and returns the model's output text
```

Self-test (offline, no pytest needed):

```
python tests/run_tests.py
```

## LLM-as-judge (day 3)

Deterministic scorers can't grade a good paraphrase. The `llm_judge`
scorer grades with a rubric prompt instead:

```
{"id": "jq-1", "input": "...", "expected": "Paris", "scorer": "llm_judge",
 "scorer_params": {"threshold": 0.8, "judge": "mock"}}
```

The prompt (`llm_evals/judge.py::render_prompt`) pins three criteria —
correctness, completeness, clarity, each 1–5 — and demands a single JSON
verdict `{"scores": {...}, "rationale": "..."}`. Verdicts that break the
contract (no JSON, out-of-range scores, missing criteria) are failed
cases, never silent passes. The criterion average is thresholded to
1.0/0.0 for the runner; the breakdown lives in the result detail.

Two judges ship. `mock` (default) is a deterministic stand-in that
grades on keyword overlap — deliberately coarse, so it runs offline.
`gemini` is the `# SWAP:` path: `scorer_params: {"judge": "gemini"}`
calls a real model behind `LLM_API_KEY`/`LLM_MODEL`. No key, no network —
it fails the case with a clear error instead of hanging.

The honest question with any judge is "does it agree with a human?".
`datasets/judge_agreement.jsonl` is a fixed 12-case labelled set
(author labels, written from the spec — stand-ins for the harness
plumbing; a real human study replaces the labels, not the code), and
`python -m llm_evals.judge` measures the gap:

```
python -m llm_evals.judge datasets/judge_agreement.jsonl
```

Measured today against the mock judge: exact agreement 0.611,
within-1 agreement **0.861**, mean absolute error 0.528, pass/fail
decision agreement **0.917**. The miss is instructive, not a bug: the
mock underrates paraphrase (agree-05) because it has no semantics —
exactly the case for swapping in the real model when labels matter.

## RAG metrics (day 4)

A RAG pipeline breaks in three places — generation, usefulness,
retrieval — so the harness scores all three (`llm_evals/rag_scorers.py`).
A RAG case carries the retrieved chunks in its input:

```json
{"id": "rag-001",
 "input": {"question": "What is the refund window?",
           "context": ["Our refund window is 30 days from purchase.", "..."]},
 "expected": "30 days from purchase, back to the original payment method.",
 "scorer": "rag_faithfulness"}
```

- `rag_faithfulness` — share of the answer's claims supported by the
  retrieved context. Claims are split per sentence (citations stripped);
  a claim is supported when most of its content tokens appear in the
  context. An abstention ("I don't know") invents nothing and scores 1.0.
- `rag_answer_relevance` — does the answer address the question?
  Deterministic stand-in: F1 of content-token overlap between answer
  and question. Crude, deterministic, honest about it.
- `rag_context_precision` — share of retrieved chunks that are
  on-point (share content tokens with the reference answer). The
  retrieval diagnostic: low precision means the retriever brought junk.

Like `llm_judge`, each metric is thresholded to 1.0/0.0 for the runner
(defaults: faithfulness 0.8, relevance 0.5, precision 0.5, overrideable
via `scorer_params.threshold`) with the metric value and breakdown in
the result detail. The mock claim check is the offline default;
`scorer_params: {"judge": "gemini"}` swaps claim-support verdicts to a
real model behind `LLM_API_KEY` (fails the case clearly with no key).

`datasets/rag_sample.jsonl` (7 cases) exercises all three against
`mock_rag_model_fn`, a naive extractive RAG model (answers from the
chunk with the most question-token overlap — grounded by construction).
The traps are hand-written via the new `fixed_output` case field, which
scores pre-recorded output without calling `model_fn` (also handy for
scoring expensive real-model runs later): an invented founding year
(faithfulness 0.50 → fail), an off-topic fluent answer (relevance 0.00
→ fail), and a 1-in-5 junk-heavy retrieval (precision 0.20 → fail).
Run it with:

```
python -m llm_evals.runner datasets/rag_sample.jsonl --model llm_evals.mock:mock_rag_model_fn
```

Measured today: 4/7 pass (the 3 traps fail by design), all 81 harness
checks green offline.

## Regression mode (day 5)

The runner already snapshots every run (`results/<dataset>/<timestamp>/`
with `cases.jsonl` + `summary.json`). `llm_evals/regression.py` diffs a
baseline against a candidate and reports what moved:

```
python -m llm_evals.regression results/sample/20261020-080000 results/sample/20261024-081500
```

```
regression: 20261020-080000 -> 20261024-081500
overall: 1.000 -> 0.900 (delta -0.100, threshold 0.0)
  contains: 1.000 -> 0.800 (delta -0.200)
newly failed (1): sample-006
no regression        # or REGRESSION — exit code 1, the CI gate
```

- Exit code: 0 when clean, 1 on regression, 2 on a bad input dir.
- `--threshold 0.02` tolerates small noise drops (default 0.0: any
  drop fails — what you want for deterministic scorers).
- Per-case lists: newly failed, fixed, output-changed-but-still-passing
  (worth eyeballing), plus new/removed case ids — which are reported
  but never count toward the verdict, since you can't regress on a
  case that didn't exist before.
- The report is also written as `regression.json` next to the
  candidate's `summary.json`.

Or gate in one command — run and compare against the checked-in
baseline at once:

```
python -m llm_evals.runner datasets/sample.jsonl --compare-with results/sample/baseline
```

That exits 1 when the run regresses, so a GitHub Action needs one
line. The runner's own exit code (1 when accuracy < 1.0) is unchanged.

Measured today: baseline-vs-baseline diffs clean (delta 0.000, exit 0),
an injected drop (3/4 → 2/4) reports `REGRESSION` with the failed case
named and exits 1; 102 harness checks green offline.

## How the runner works (days 2–7)

```
datasets/*.jsonl ──▶ runner ──▶ model_fn(input) ──▶ scorers ──▶ results/
                                                        │
                                    snapshots/*.json + report.md
```

- **Day 2** — dataset runner: JSONL cases → `model_fn` → scored results,
  per-case JSONL + aggregate table.
- **Day 3** — LLM-as-judge scorer: rubric prompt, mock judge default
  (offline), real-model judge behind `# SWAP:`, human-agreement check
  on a fixed labelled set.
- **Day 4** — RAG scorers: faithfulness (claim-level support),
  answer relevance, context precision — the three that actually
  diagnose retrieval vs generation failures.
- **Day 5** — regression mode: diffs a candidate run against the baseline
  snapshot and exits 1 on accuracy drops — the CI gate
  (`llm_evals/regression.py`, or one-command via
  `runner.py --compare-with <baseline>`).
- **Day 6** — worked examples: the rag-service golden set and the
  agent-service task set wired in as example datasets, with their
  real mock `model_fn`s.
- **Day 7** — GitHub Action running the harness on push + release notes.

## Layout

```
llm-evals/
  README.md            this file: philosophy + format spec
  datasets/            golden JSONL datasets (the product)
  llm_evals/
    runner.py          day 2: dataset → model_fn → scored results
    scorers.py         days 2-3: exact / contains / regex / json_schema / llm_judge
    judge.py           day 3: LLM-as-judge + agreement check
    rag_scorers.py     day 4: faithfulness / relevance / precision
    regression.py      day 5: snapshot diff + CI gate
  snapshots/           checked-in score snapshots (CI compares against these)
  .github/workflows/   day 7: evals on push
```

## Non-goals

- Not a prompt playground, not a chat UI. There are enough of those.
- Not a leaderboard. Scores are for *your* runs over time, not for
  comparing models against each other.
- No hosted service. It's a library + CLI you run in CI.

## License

MIT — evals are infrastructure, infrastructure should be boring and free.
