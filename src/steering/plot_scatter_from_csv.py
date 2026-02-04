# src/steering/plot_scatter_from_csv.py
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def prompt_label(prompt_type: str) -> str:
    # Map internal labels to paper-friendly labels
    return "Greater-Than prompts" if prompt_type == "clean" else "Less-Than prompts"


def dir_label(direction: str) -> str:
    return "+α" if direction == "plus" else "−α"


def signed_alpha(alpha_mag: float, direction: str) -> float:
    d = str(direction).lower()
    if d == "minus":
        return -abs(float(alpha_mag))
    if d == "plus":
        return abs(float(alpha_mag))
    return float(alpha_mag)


def safe_fname(s: str) -> str:
    # Simple filename sanitizer
    return (
        s.replace(" ", "_")
        .replace("|", "_")
        .replace("α", "alpha")
        .replace("+", "p")
        .replace("-", "m")
        .replace("−", "m")
    )


def scatter(df: pd.DataFrame, title: str, outpath: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(df["base_margin"], df["steered_margin"], s=10, alpha=0.6)

    # identity line y=x
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
    ap.add_argument("--csv", type=str, required=True, help="points CSV from eval script")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory for scatters")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    required = {"base_margin", "steered_margin", "prompt_type", "direction", "layer", "head", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Four conditions: clean/corrupt × plus/minus
    for ptype in ["clean", "corrupt"]:
        for direction in ["plus", "minus"]:
            sub = df[(df["prompt_type"] == ptype) & (df["direction"] == direction)]
            if len(sub) == 0:
                continue

            alpha_mag = float(sub["alpha"].iloc[0])
            a_signed = signed_alpha(alpha_mag, direction)
            layer = int(sub["layer"].iloc[0])
            head = int(sub["head"].iloc[0])

            title = (
                f"{prompt_label(ptype)} | "
                f"Layer{layer}Head{head} | "
                f"{dir_label(direction)} (α={a_signed:+.0f})"
            )

            fname = safe_fname(
                f"scatter_{prompt_label(ptype)}_{dir_label(direction)}_Layer{layer}Head{head}_a{a_signed:+.0f}.png"
            )
            outpath = os.path.join(args.out_dir, fname)

            scatter(sub, title, outpath)

    print("Saved scatter plots to:", args.out_dir)


if __name__ == "__main__":
    main()
