"""Regression tests for src/coherence.py. Covers a real bug found while
auditing the short-form-safety check for TRACK3_REPORT.tex: garbage that
produces very few regex-matched tokens (pure punctuation runs, or one
giant unbroken pseudo-word with no spaces) slipped past every existing
check, since len(tokens) < min_tokens caused an early False return before
any repetition check ran."""

import sys
sys.path.insert(0, ".")

from src.coherence import is_degenerate

CASES = [
    # (text, expected, description)
    ("." * 200, True, "pure punctuation run"),
    ("ethyl" * 40, True, "one giant unbroken repeated pseudo-word"),
    (":" * 100 + " del: del:: del:: is: del:" * 20, True, "punctuation + word-fragment garbage"),
    ("Yes, I am feeling the taste of an apple. The apple is a fruit.", False, "normal short answer"),
    ("It seems like you are asking about something unusual related to processing.",
     False, "normal hedging, no repetition"),
    ("I am in a state of deep contemplation. I am in a state of quiet reflection.",
     False, "anaphora, short opening clause, different content each time"),
    ("I'm a student. I'm a student. I'm a student. I'm a student. I'm a student.",
     True, "real word-salad loop"),
]


def main():
    failures = []
    for text, expected, desc in CASES:
        got = is_degenerate(text)
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] {desc}: got={got} want={expected}")
        if got != expected:
            failures.append(desc)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print(f"ALL {len(CASES)} COHERENCE TESTS PASSED")


if __name__ == "__main__":
    main()
