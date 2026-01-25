# src/steering/plot_scatter_from_csv.py
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def scatter(df, title, outpath):
    plt.figure()
    plt.scatter(df["base_margin"], df["steered_margin"], s=10, alpha=0.6)
    # identity line
    lo = min(df["base_margin"].min(), df["steered_margin"].min())
    hi = max(df["base_margin"].max(), df["steered_margin"].max())
    plt.plot([lo, hi], [lo, hi], "--", color="gray")
    plt.title(title)
    plt.xlabel("base_margin = logp(Yes) - logp(No)")
    plt.ylabel("steered_margin = logp(Yes) - logp(No)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="../../outputs/clean_corrupt_steering_points.csv")
    ap.add_argument("--out_dir", type=str, default="../../outputs/scatters")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Four conditions: clean/corrupt × plus/minus
    for prompt_type in ["clean", "corrupt"]:
        for direction in ["plus", "minus"]:
            sub = df[(df["prompt_type"] == prompt_type) & (df["direction"] == direction)]
            if len(sub) == 0:
                continue
            alpha = sub["alpha"].iloc[0]
            layer = sub["layer"].iloc[0]
            head = sub["head"].iloc[0]
            title = f"{prompt_type} | {direction} | L{layer} H{head} α={alpha}"
            outpath = os.path.join(args.out_dir, f"scatter_{prompt_type}_{direction}.png")
            scatter(sub, title, outpath)

    print("Saved scatter plots to:", args.out_dir)


if __name__ == "__main__":
    main()
