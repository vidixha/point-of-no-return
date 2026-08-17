"""Unit tests for the thinking/final split (src/thinking_split.py) used by
the experiment runner and the removal-test / entropy-trace Modal
functions."""

import sys
sys.path.insert(0, ".")

from src.thinking_split import split_thinking_and_final


def test_split():
    cases = [
        # standard case: prompt already seeded opening <think>, generated text
        # is reasoning + </think> + final answer.
        ("I keep thinking about apples for some reason.</think>\n\nNo, nothing unusual.",
         "I keep thinking about apples for some reason.", "No, nothing unusual.", True),
        # defensive case: an opening <think> tag present in generated text too.
        ("<think>reasoning about apple here</think>final answer here",
         "reasoning about apple here", "final answer here", True),
        # truncated: ran out of max_new_tokens before closing the think block.
        ("still reasoning and reasoning with no end in sight",
         "still reasoning and reasoning with no end in sight", "", False),
        # Gemma 4's different marker convention.
        ("<|channel>thought\nthinking about apples<channel|>No, nothing unusual.",
         "thinking about apples", "No, nothing unusual.", True),
    ]
    failures = []
    for raw, want_think, want_final, want_complete in cases:
        think, final, complete = split_thinking_and_final(raw)
        ok = think == want_think and final == want_final and complete == want_complete
        status = "OK" if ok else "FAIL"
        print(f"[{status}] raw={raw[:50]!r} -> think={think!r} final={final!r} complete={complete}")
        if not ok:
            failures.append(raw)
    return failures


def main():
    failures = test_split()
    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        sys.exit(1)
    print("ALL THINKING_SPLIT TESTS PASSED")


if __name__ == "__main__":
    main()
