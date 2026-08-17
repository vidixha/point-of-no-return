"""Generates the critical-window figure for all seven models: valid
completion (recovered vs not) as a function of k, the number of tokens
the concept vector was injected for before removal. Computed directly
from the saved removal-test JSON files (not hand-typed numbers).

Several models share an identical binary recovery trajectory (e.g.
Qwen3-1.7B and Qwen3-8B), which would otherwise hide one line under
another. Each model gets a small fixed vertical offset (visual only,
noted in the caption) plus a distinct linestyle so every trajectory
stays visible. The legend sits outside the plot area so it never
overlaps a data line."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = [
    ("results/knockout_qwen3_1p7b_apple.json", "Qwen3-1.7B", "#4C72B0", "-"),
    ("results/knockout_qwen3_4b_apple.json", "Qwen3-4B", "#DD8452", "--"),
    ("results/knockout_qwen3_8b_apple.json", "Qwen3-8B", "#55A868", "-."),
    ("results/knockout_deepseek_r1_distill_qwen_7b_apple.json", "DS-R1-Distill-Qwen-7B", "#C44E52", ":"),
    ("results/knockout_deepseek_r1_distill_llama_8b_apple.json", "DS-R1-Distill-Llama-8B", "#8172B2", "-"),
    ("results/knockout_phi4_mini_reasoning_apple.json", "Phi-4-mini", "#937860", "--"),
    ("results/knockout_gemma4_12b_apple.json", "Gemma 4 12B", "#CCB974", "-."),
]

OFFSET = 0.022  # small fixed vertical jitter per model, visual only


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for i, (path, label, color, ls) in enumerate(FILES):
        with open(path) as f:
            d = json.load(f)
        ks = [c["knockout_k"] for c in d["conditions"]]
        valid = [1 if c["valid_completion"] else 0 for c in d["conditions"]]
        order = sorted(range(len(ks)), key=lambda i: ks[i])
        ks = [ks[i] for i in order]
        valid = [valid[i] for i in order]
        offset = (i - (len(FILES) - 1) / 2) * OFFSET
        valid_j = [v + offset for v in valid]
        ax.plot(ks, valid_j, marker="o", markersize=8, linewidth=2.2,
                 linestyle=ls, color=color, label=label)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("$k$: tokens injected before the vector is removed", fontsize=13)
    ax.set_ylabel("Recovers to a valid completion", fontsize=13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xlim(0.7, 1000)
    ax.legend(fontsize=11.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               frameon=False, ncol=1, borderaxespad=0)
    ax.set_title("Recovery after removal at token $k$, by model", fontsize=14)
    fig.text(0.5, -0.02,
              "Lines are offset vertically for visibility only; recovery is either yes or no.",
              ha="center", fontsize=10, color="#555555")
    fig.tight_layout()
    fig.savefig("plots/critical_window.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/critical_window.png")


if __name__ == "__main__":
    main()
