import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def signed_alpha(alpha_mag: float, direction: str) -> float:
    d = str(direction).lower()
    if d == "minus":
        return -abs(float(alpha_mag))
    if d == "plus":
        return abs(float(alpha_mag))
    return float(alpha_mag)


def plot_panel(ax, df_panel, title, colors):
    # Plot both directions on same axes
    for direction in ["minus", "plus"]:
        g = df_panel[df_panel["direction"] == direction]
        if len(g) == 0:
            continue
        ax.scatter(
            g["base_margin"].to_numpy(),
            g["steered_margin"].to_numpy(),
            s=18,
            alpha=0.75,
            label=f"{direction} (α={signed_alpha(g['alpha'].iloc[0], direction):+.1f})",
            color=colors[direction],
        )

    # y=x reference line
    x = df_panel["base_margin"].to_numpy()
    y = df_panel["steered_margin"].to_numpy()
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("base_margin = logp(Yes) - logp(No)")
    ax.set_ylabel("steered_margin = logp(Yes) - logp(No)")
    ax.legend(frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="points CSV from eval script")
    ap.add_argument("--out", required=True, help="output .png path")
    ap.add_argument("--title", default=None, help="overall title")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"base_margin", "steered_margin", "prompt_type", "direction", "layer", "head", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Fixed colors as requested
    colors = {"minus": "tab:blue", "plus": "tab:orange"}

    # Identify (layer, head) for labeling
    layer = int(df["layer"].iloc[0])
    head = int(df["head"].iloc[0])
    alpha_mag = float(df["alpha"].iloc[0])

    clean = df[df["prompt_type"] == "clean"].copy()
    corrupt = df[df["prompt_type"] == "corrupt"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    plot_panel(
        axes[0],
        clean,
        title=f"clean | L{layer} H{head} | |α|={alpha_mag}",
        colors=colors,
    )

    plot_panel(
        axes[1],
        corrupt,
        title=f"corrupt | L{layer} H{head} | |α|={alpha_mag}",
        colors=colors,
    )

    if args.title:
        fig.suptitle(args.title, fontsize=14)

    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
