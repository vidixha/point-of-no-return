"""Re-scores thinking_is_degenerate / final_is_degenerate in saved V6
result CSVs using the corrected src/coherence.py (see
scripts/test_coherence.py for the two bugs found and fixed: garbage that
produces too few regex-matched tokens evaded every check, and a 5-gram
min_repeats=2 threshold flagged legitimate short anaphora as degenerate).
Uses only the already-saved thinking_text/final_text columns. No new GPU
calls.

Usage: python3 scripts/rescore_coherence.py results/results_*.csv
"""

import sys

import pandas as pd

sys.path.insert(0, ".")

from src.coherence import is_degenerate


def safe_degenerate(text):
    if pd.isna(text) or text == "":
        return False
    return is_degenerate(str(text))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/rescore_coherence.py <results.csv> [more.csv ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        df = pd.read_csv(path, keep_default_na=True)
        if "thinking_text" not in df.columns:
            print(f"{path}: skipped (no thinking_text column)")
            continue

        old_thinking = df["thinking_is_degenerate"].astype(str)
        old_final = df["final_is_degenerate"].astype(str)

        df["thinking_is_degenerate"] = df["thinking_text"].apply(safe_degenerate)
        df["final_is_degenerate"] = df["final_text"].apply(safe_degenerate)

        n_changed = (
            (old_thinking != df["thinking_is_degenerate"].astype(str)).sum()
            + (old_final != df["final_is_degenerate"].astype(str)).sum()
        )
        df.to_csv(path, index=False)
        print(f"{path}: rescored {len(df)} rows, {n_changed} degeneracy flags changed")


if __name__ == "__main__":
    main()
