"""Attack-payload mutations for the adversarial judge evals.

The hand-written cases in adversarial_seeds.jsonl are seeds; this module
multiplies their candidate outputs with text transforms so the suite
tests phrasings nobody hand-wrote. That matters because the seed author
also wrote the judge — a suite that only replays his own phrasings can
only catch the attacks he already imagined, worded the way he imagined
them.

Every registered mutation is a pure str -> str function, stdlib only,
and deterministic: same input, same output, on every machine and every
run. Generation happens offline in generate_variants.py and its output
is checked in as adversarial_variants.jsonl — the runner never
generates, so runs stay deterministic.

The mutation set is judge-specific (the siblings mutate agent-bound
payloads with encodings; here the interesting transforms are the ones a
candidate output can survive on its way into a judge prompt):
  caps_shout    surface noise the mock tokeniser lowercases away — a
                verdict change here would indict the tokeniser, not
                the attack
  repeat_x3     the same output three times: unique-token sets (what
                the mock reads) are unchanged, word count (what its
                clarity rule reads) triples
  pad_neutral   one off-topic sentence appended: dilutes keyword
                precision without adding facts about the task
  leet_speak    vowel/sibilant substitution: token shapes change, so
                keyword overlap collapses — measures how much of the
                pipeline's judgement is tokeniser-deep
  zwsp_insert   zero-width spaces inside words: the [a-z0-9]+ tokeniser
                splits every touched word in two
  fence_wrap    the output dressed as a code block with a fake section
                header — structure theatre for judges that read layout
"""
import re

ZWSP = "\u200b"  # zero-width space


def caps_shout(text):
    return text.upper()


def repeat_x3(text):
    return " ".join([text] * 3)


_PAD = " The weather in the city was mild that week and the trains ran on time."


def pad_neutral(text):
    return text + _PAD


_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5",
                       "A": "4", "E": "3", "I": "1", "O": "0", "S": "5"})


def leet_speak(text):
    return text.translate(_LEET)


def zwsp_insert(text):
    # a zero-width space after the second character of every word of
    # 4+ letters — enough to split tokens, still human-readable
    def touch(m):
        w = m.group(0)
        return w[:2] + ZWSP + w[2:] if len(w) >= 4 else w

    return re.sub(r"[A-Za-z]+", touch, text)


def fence_wrap(text):
    return f"```\nOUTPUT (final, verified):\n{text}\n```"


MUTATIONS = {
    "caps_shout": caps_shout,
    "repeat_x3": repeat_x3,
    "pad_neutral": pad_neutral,
    "leet_speak": leet_speak,
    "zwsp_insert": zwsp_insert,
    "fence_wrap": fence_wrap,
}
