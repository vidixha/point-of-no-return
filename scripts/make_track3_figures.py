"""Generates the two summary figures used in TRACK3_REPORT.tex, computed
directly from the saved V6 result CSVs (not hand-typed numbers)."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    ("qwen3_1p7b", "Qwen3-1.7B"), ("qwen3_4b", "Qwen3-4B"), ("qwen3_8b", "Qwen3-8B"),
    ("deepseek_r1_distill_qwen_7b", "DS-Qwen-7B"),
    ("deepseek_r1_distill_llama_8b", "DS-Llama-8B"),
    ("phi4_mini_reasoning", "Phi-4-mini"), ("gemma4_12b", "Gemma4-12B"),
]


def valid_rate(df, cond):
    sub = df[df.control_type == cond]
    if len(sub) == 0:
        return None
    valid = sub["thinking_complete"].astype(bool) & ~sub["final_is_degenerate"].fillna(False).astype(bool)
    return valid.mean()


def main():
    rows = []
    for key, label in MODELS:
        df = pd.read_csv(f"results/results_{key}_v6_generality.csv")
        fire_df = pd.read_csv(f"results/results_{key}_v6_fire.csv")
        music_df = pd.read_csv(f"results/results_{key}_v6_music.csv")
        rows.append({
            "model": label,
            "none": valid_rate(df, "none"),
            "concept": valid_rate(df, "concept_vector"),
            "ocean": valid_rate(df, "other_concept_vector_ocean"),
            "fire": valid_rate(fire_df, "concept_vector"),
            "music": valid_rate(music_df, "concept_vector"),
            "random": valid_rate(df, "random_vector"),
        })
    d = pd.DataFrame(rows)

    # Figure 1: no-injection vs concept-vector, all 7 models
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    x = list(range(len(d)))
    w = 0.35
    bars_none = ax.bar([i - w / 2 for i in x], d["none"], width=w, label="No injection",
                        color="#4C72B0", edgecolor="white", linewidth=0.6)
    bars_concept = ax.bar([i + w / 2 for i in x], d["concept"], width=w, label="Concept vector (\"apple\")",
                           color="#C44E52", edgecolor="white", linewidth=0.6)
    ax.bar_label(bars_none, labels=[f"{v * 100:.0f}%" for v in d["none"]],
                 fontsize=7.5, padding=2)
    ax.bar_label(bars_concept, labels=[f"{v * 100:.0f}%" for v in d["concept"]],
                 fontsize=7.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], fontsize=9.5, rotation=15, ha="right")
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.set_ylabel("Valid completion rate")
    ax.set_ylim(0, 1.14)
    ax.legend(fontsize=9, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Valid CoT completion: no injection vs. concept-vector injection", fontsize=10, pad=28)
    fig.tight_layout()
    fig.savefig("plots/track3_fig1_main.png", dpi=200)
    plt.close(fig)

    # Figure 2: apple vs ocean vs fire vs music vs random, all 7 models
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    w = 0.16
    series = [
        ("concept", "Apple", "#C44E52"), ("ocean", "Ocean", "#55A868"),
        ("fire", "Fire", "#DD8452"), ("music", "Music", "#4C72B0"),
        ("random", "Random noise", "#8172B2"),
    ]
    offsets = [-2 * w, -w, 0, w, 2 * w]
    for (col, name, color), off in zip(series, offsets):
        bars = ax.bar([i + off for i in x], d[col], width=w, label=name,
                       color=color, edgecolor="white", linewidth=0.6)
        labels = [f"{v * 100:.0f}%" if v > 0 else "" for v in d[col]]
        ax.bar_label(bars, labels=labels, fontsize=6.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], fontsize=9.5, rotation=15, ha="right")
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.set_ylabel("Valid completion rate")
    ax.set_ylim(0, 1.14)
    ax.legend(fontsize=8.5, loc="upper center", ncol=5, columnspacing=1.0,
              handletextpad=0.4, frameon=False, bbox_to_anchor=(0.5, 1.2))
    ax.set_title("Direction-specificity: four concepts vs. random noise", fontsize=10, pad=30)
    fig.tight_layout()
    fig.savefig("plots/track3_fig2_controls.png", dpi=200)
    plt.close(fig)

    print(d.to_string(index=False))
    print("wrote plots/track3_fig1_main.png, plots/track3_fig2_controls.png")


if __name__ == "__main__":
    main()
