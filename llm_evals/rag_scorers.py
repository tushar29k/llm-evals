"""RAG metrics: faithfulness, answer relevance, context precision.

The three metrics that actually diagnose where a RAG pipeline breaks —
generation (faithfulness), usefulness (answer relevance), or retrieval
(context precision). A RAG case carries the retrieved chunks in its input:

    {"id": "rag-001",
     "input": {"question": "...", "context": ["chunk 1", "chunk 2"]},
     "expected": "reference answer",
     "scorer": "rag_faithfulness"}

All three ship deterministic stand-ins (token-overlap based) behind the
scorer names rag_faithfulness / rag_answer_relevance / rag_context_precision.
The stand-ins are coarse on purpose — paraphrase is their blind spot, same
as the day-3 judge. The real-model variants live behind scorer_params
{"judge": "gemini"} via the same # SWAP: seam as judge.py.

Scores follow the llm_judge convention: the continuous metric is thresholded
(scorer_params.threshold, per-metric defaults below) to 1.0/0.0 for the
runner, and the metric value + breakdown always ride in detail.
"""

import json
import os
import re

from llm_evals.judge import JudgeConfigError

# token match rules, shared by the three metrics
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or but if then else when where what which who whom "
    "this that these those is are was were be been being has have had do "
    "does did will would can could should may might must of in on at to "
    "for with by from as it its we you they he she my your our their not "
    "no yes so such than too very about into over after before between "
    .split()
)
# overlap fraction of a claim's content tokens that must appear in the
# context for the claim to count as supported (mock only)
SUPPORT_FRAC = 0.5
# overlap fraction of a chunk's content tokens shared with the reference
# for the chunk to count as retrieved-on-point (mock only)
CHUNK_RELEVANT_FRAC = 0.2
# claims like these are abstentions, not hallucinations
_ABSTAIN = ("i don't know", "do not know", "no information", "not enough information",
            "cannot answer", "can't answer", "no relevant")

DEFAULT_THRESHOLDS = {
    "rag_faithfulness": 0.8,
    "rag_answer_relevance": 0.5,
    "rag_context_precision": 0.5,
}


def _content_tokens(text):
    return [t for t in _TOKEN.findall(str(text).lower()) if t not in _STOP]


def split_claims(answer):
    """One checkable claim per sentence. Citations stripped, fragments dropped."""
    text = re.sub(r"\[[^\]]*\]", " ", str(answer))  # [1], [source], ...
    text = re.sub(r"\((?:source|src|citation)[^)]*\)", " ", text, flags=re.I)
    claims = []
    for sent in re.split(r"[.!?\n]+", text):
        sent = sent.strip()
        if len(_content_tokens(sent)) >= 3:  # below this there's nothing to check
            claims.append(sent)
    return claims


def _is_abstention(claim):
    low = claim.lower()
    return any(phrase in low for phrase in _ABSTAIN)


def mock_claim_supported(claim, context_text):
    """Deterministic stand-in for "does the context support this claim?"."""
    if _is_abstention(claim):
        return True  # an abstention invents nothing — faithful by construction
    claim_toks = set(_content_tokens(claim))
    ctx_toks = set(_content_tokens(context_text))
    if not claim_toks:
        return True
    return len(claim_toks & ctx_toks) / len(claim_toks) >= SUPPORT_FRAC


def gemini_claim_checker(api_key=None, model=None):
    """# SWAP: real-model claim-support verdicts, same verdict contract.

    Raises JudgeConfigError without a key — the deterministic check stays
    the offline default.
    """
    import urllib.request

    key = api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise JudgeConfigError("LLM_API_KEY not set — deterministic checks stay the default")
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

    def supported(claim, context_text):
        verdict = call(
            "Does the CONTEXT support the CLAIM? Answer with exactly YES or NO.\n\n"
            f"CLAIM: {claim}\n\nCONTEXT:\n{context_text}"
        )
        return verdict.strip().upper().startswith("Y")

    return supported


def faithfulness(answer, context, judge="mock"):
    """Share of answer claims supported by the retrieved context.

    Returns (metric, detail). No checkable claims (pure abstention) is 1.0 —
    saying nothing beats inventing something.
    """
    claims = split_claims(answer)
    if not claims:
        return 1.0, "no checkable claims (abstention) — faithful by construction"
    context_text = "\n".join(context)
    if judge == "gemini":
        supported_fn = gemini_claim_checker()
        verdicts = [supported_fn(c, context_text) for c in claims]
    else:
        verdicts = [mock_claim_supported(c, context_text) for c in claims]
    n_ok = sum(verdicts)
    metric = n_ok / len(claims)
    missed = [c for c, ok in zip(claims, verdicts) if not ok]
    detail = f"{n_ok}/{len(claims)} claims supported"
    if missed:
        detail += " — unsupported: " + "; ".join(m[:80] for m in missed)
    return round(metric, 3), detail


def answer_relevance(question, answer):
    """Does the answer address the question? Mock: F1 of content-token
    overlap between answer and question. Crude, deterministic, honest about it."""
    q_toks = set(_content_tokens(question))
    a_toks = set(_content_tokens(answer))
    overlap = q_toks & a_toks
    if not q_toks or not a_toks:
        return 0.0, "empty question or answer"
    precision = len(overlap) / len(a_toks)
    recall = len(overlap) / len(q_toks)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    detail = f"token F1 {f1:.2f} (p {precision:.2f}, r {recall:.2f}, {len(overlap)} shared terms)"
    return round(f1, 3), detail


def context_precision(context, reference):
    """Share of retrieved chunks that are on-point. The retrieval diagnostic:
    low precision = the retriever brought junk."""
    chunks = list(context)
    if not chunks:
        return 0.0, "no chunks retrieved"
    ref_toks = set(_content_tokens(reference))
    relevant = 0
    for chunk in chunks:
        c_toks = set(_content_tokens(chunk))
        if c_toks and ref_toks and len(c_toks & ref_toks) / len(c_toks) >= CHUNK_RELEVANT_FRAC:
            relevant += 1
    return round(relevant / len(chunks), 3), f"{relevant}/{len(chunks)} chunks on-point"


def _rag_inputs(case):
    # one input shape for all three metrics: {"question": ..., "context": [...]}
    payload = case.get("input")
    if not isinstance(payload, dict) or "question" not in payload or "context" not in payload:
        raise ValueError(
            f"case {case['id']}: rag scorers need "
            "input = {'question': ..., 'context': [...chunks...]}"
        )
    if not isinstance(payload["context"], list):
        raise ValueError(f"case {case['id']}: input['context'] must be a list of chunks")
    return payload["question"], payload["context"]


def _threshold(case, name):
    params = case.get("scorer_params") or {}
    return params.get("threshold", DEFAULT_THRESHOLDS[name])


def _as_runner_score(name, metric, detail, threshold):
    passed = 1.0 if metric >= threshold else 0.0
    return passed, f"{name} {metric:.2f} {'≥' if passed else '<'} {threshold} → {'pass' if passed else 'fail'} · {detail}"


def _rag_faithfulness(case, output):
    question, context = _rag_inputs(case)
    params = case.get("scorer_params") or {}
    try:
        metric, detail = faithfulness(output, context, judge=params.get("judge", "mock"))
    except Exception as e:  # noqa — a broken judge is a failed case, not a dead run
        return 0.0, f"judge error: {type(e).__name__}: {str(e)[:120]}"
    return _as_runner_score("faithfulness", metric, detail, _threshold(case, "rag_faithfulness"))


def _rag_answer_relevance(case, output):
    question, context = _rag_inputs(case)
    metric, detail = answer_relevance(question, output)
    return _as_runner_score("answer_relevance", metric, detail,
                            _threshold(case, "rag_answer_relevance"))


def _rag_context_precision(case, output):
    question, context = _rag_inputs(case)
    metric, detail = context_precision(context, case["expected"])
    return _as_runner_score("context_precision", metric, detail,
                            _threshold(case, "rag_context_precision"))


DISPATCH = {
    "rag_faithfulness": _rag_faithfulness,
    "rag_answer_relevance": _rag_answer_relevance,
    "rag_context_precision": _rag_context_precision,
}
