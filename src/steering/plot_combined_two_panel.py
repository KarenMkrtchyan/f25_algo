# src/steering/plot_combined_two_panel.py
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def signed_alpha(alpha_mag: float, direction: str) -> float:
    return -abs(alpha_mag) if direction == "minus" else abs(alpha_mag)


def plot_panel(ax, df_panel, panel_title: str):
    colors = {"minus": "tab:blue", "plus": "tab:orange"}

    for direction in ["minus", "plus"]:
        g = df_panel[df_panel["direction"] == direction]
        if len(g) == 0:
            continue
        a = float(g["alpha"].iloc[0])
        ax.scatter(
            g["base_margin"].to_numpy(),
            g["steered_margin"].to_numpy(),
            s=18,
            alpha=0.75,
            color=colors[direction],
            label=f"{direction} (α={signed_alpha(a, direction):+.0f})",
        )

    x = df_panel["base_margin"].to_numpy()
    y = df_panel["steered_margin"].to_numpy()
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(panel_title)
    ax.set_xlabel("base_margin = logp(Yes) - logp(No)")
    ax.set_ylabel("steered_margin = logp(Yes) - logp(No)")
    ax.legend(frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.csv)

    required = {"base_margin", "steered_margin", "prompt_type", "direction", "layer", "head", "alpha", "model_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    layer = int(df["layer"].iloc[0])
    head = int(df["head"].iloc[0])
    alpha_mag = float(df["alpha"].iloc[0])
    model_id = str(df["model_id"].iloc[0])

    left = df[df["prompt_type"] == "original"].copy()
    right = df[df["prompt_type"] == "swapped"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plot_panel(axes[0], left, f"original | Layer{layer}Head{head} | α=±{int(alpha_mag)}")
    plot_panel(axes[1], right, f"swapped | Layer{layer}Head{head} | α=±{int(alpha_mag)}")

    if args.title is not None:
        fig.suptitle(args.title, fontsize=14)
    else:
        fig.suptitle(f"{model_id} | Layer{layer}Head{head} | α=±{int(alpha_mag)}", fontsize=14)

    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
