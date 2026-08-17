"""Cheap coherence/degeneration heuristic for generated text (TASK_V2.md:
"record ... repetition/degeneration, generation coherence"). Not a language
model judge -- just flags obviously broken output (the "No. No. No." /
token-salad failure mode seen in the V1 alpha=2.0 runs) so it can be
tabulated alongside the other metrics.
"""

import re
from collections import Counter


# Keeps English contractions ("I'm", "don't") as single tokens and CJK text
# as one character per token (no whitespace to split on). The earlier
# \w+-only version split "I'm" into "i" + "m", which inflated bigram counts
# for ordinary text that just uses contractions a few times and produced
# false degenerate flags on fine, coherent responses.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[一-鿿]")


def is_degenerate(text, min_tokens=8, repeat_ratio_threshold=0.4):
    """True if a small number of tokens dominate the response (word-salad
    loop like "No. No. No."), or if a substantial phrase (4+ words) repeats
    verbatim -- the hallmark of an activation-steering breakdown, as
    distinct from ordinary stylistic repetition (anaphora reusing a short
    opening clause with different content each time, e.g. "I am in a state
    of X... I am in a state of Y...", which is NOT flagged)."""
    text = text.strip()
    if not text:
        return False

    # Catch garbage that produces few or no regex-matched tokens: pure
    # punctuation runs ("......."), or one giant unbroken pseudo-word with
    # no spaces ("ethylethylethyl..."). These evade every check below,
    # since len(tokens) can be 0 (no letters at all) or 1 (one huge
    # "token" with no internal spaces), both under min_tokens. Check
    # directly on the raw text for a short substring (1-15 chars)
    # immediately repeating 5+ times in a row.
    if re.search(r"(.{1,15}?)\1{4,}", text):
        return True

    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < min_tokens:
        return False

    counts = Counter(t.lower() for t in tokens)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(tokens) >= repeat_ratio_threshold:
        return True

    # exact repeated LONG phrase (4-5 tokens) -- short 2-3 word repeats are
    # common in ordinary prose (anaphora, filler); a true generation loop
    # repeats a much longer span near-verbatim. min_repeats=3 for both
    # lengths: a 5-word opening clause repeating just twice with different
    # content after ("I am in a state of X... I am in a state of Y...") is
    # the docstring's own canonical example of legitimate anaphora, and a
    # min_repeats=2 threshold at n=5 flagged that exact case as degenerate,
    # contradicting the stated design intent. Found via
    # scripts/test_coherence.py.
    for n, min_repeats in ((4, 3), (5, 3)):
        if len(tokens) < n * 2:
            continue
        ngrams = [" ".join(tokens[i:i + n]).lower() for i in range(len(tokens) - n + 1)]
        ngram_counts = Counter(ngrams)
        if ngram_counts and ngram_counts.most_common(1)[0][1] >= min_repeats:
            return True

    return False
