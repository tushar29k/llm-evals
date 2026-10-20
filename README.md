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
| `scorer` | yes | Which scorer grades this case: `exact`, `contains`, `regex`, `json_schema`, `llm_judge`. |
| `scorer_params` | no | Scorer-specific knobs, e.g. `{"pattern": "..."}` for regex, `{"schema": {...}}` for json_schema, `{"rubric": "..."}` for llm_judge. |
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
- **Day 5** — regression mode: compare two snapshots, fail (exit 1)
  on any aggregate drop beyond tolerance. This is the CI gate.
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
    scorers.py         day 2: exact / contains / regex / json_schema
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
