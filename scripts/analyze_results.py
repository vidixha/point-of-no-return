"""Analysis across all models tested.

Uses a corrected "genuinely_valid" completion metric rather than raw
thinking_complete: Gemma 4 12B's concept_vector condition revealed that a
model can close its reasoning-block marker (satisfying a naive
thinking_complete=True check) while still being fully degenerate --
spamming the literal formatting tokens themselves ("<|channel>thought",
"thought thought thought...") rather than producing a real answer.
genuinely_valid = thinking_complete AND NOT final_is_degenerate catches
this. Applied uniformly to all models for an apples-to-apples table (most
models' numbers are unaffected; Qwen3-4B and Qwen3-8B's random_vector
numbers change meaningfully; Gemma 4's concept_vector number flips from a
misleading 100% to the correct 0%).

Usage: python3 scripts/analyze_results.py
"""

import math

import pandas as pd

MODELS = [
    "qwen3_1p7b", "qwen3_4b", "qwen3_8b",
    "deepseek_r1_distill_qwen_7b", "deepseek_r1_distill_llama_8b",
    "phi4_mini_reasoning", "gemma4_12b",
]
CONDITIONS = ["none", "concept_vector", "random_vector"]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    rows = []
    for m in MODELS:
        df = pd.read_csv(f"results/results_{m}_v6_generality.csv")
        df["genuinely_valid"] = df["thinking_complete"].astype(bool) & ~df["final_is_degenerate"].fillna(False).astype(bool)
        for cond in CONDITIONS:
            sub = df[df["control_type"] == cond]
            n = len(sub)
            if n == 0:
                continue
            k_raw = int(sub["thinking_complete"].astype(bool).sum())
            k_valid = int(sub["genuinely_valid"].sum())
            lo, hi = wilson_ci(k_valid, n)
            rows.append({
                "model": m, "condition": cond, "n": n,
                "raw_completion_rate": round(k_raw / n, 3),
                "valid_completion_rate": round(k_valid / n, 3),
                "valid_ci_lo": round(lo, 3), "valid_ci_hi": round(hi, 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv("results/metrics_summary_v6.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
