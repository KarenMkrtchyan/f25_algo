import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def signed_alpha(alpha_mag: float, direction: str) -> float:
    direction = str(direction).lower()
    if direction in ("minus", "neg", "negative", "-"):
        return -abs(float(alpha_mag))
    if direction in ("plus", "pos", "positive", "+"):
        return abs(float(alpha_mag))
    # fallback: unknown direction, keep magnitude
    return float(alpha_mag)


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plots for Qwen-3-1.7B steering experiments"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to points CSV")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix (e.g. 'Qwen3-1.7B | L17 H1')",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    required = {"base_margin", "steered_margin", "prompt_type", "direction", "layer", "head", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for (prompt_type, direction), g in df.groupby(["prompt_type", "direction"]):
        layer = int(g["layer"].iloc[0])
        head = int(g["head"].iloc[0])
        alpha_mag = float(g["alpha"].iloc[0])
        a_signed = signed_alpha(alpha_mag, direction)

        x = g["base_margin"].to_numpy()
        y = g["steered_margin"].to_numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, alpha=0.7)

        # y = x reference line
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        plt.plot([lo, hi], [lo, hi], "--", color="gray")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)

        plt.xlabel("base_margin = logp(Yes) - logp(No)")
        plt.ylabel("steered_margin = logp(Yes) - logp(No)")

        if args.title:
            title = f"{args.title} | {prompt_type} | α={a_signed:+.1f}"
        else:
            title = f"Qwen3-1.7B | L{layer} H{head} | {prompt_type} | α={a_signed:+.1f}"

        plt.title(title)

        # filename includes signed alpha
        out_path = os.path.join(
            args.out_dir,
            f"scatter_{prompt_type}_L{layer}_H{head}_a{a_signed:+.1f}.png".replace("+", "p").replace("-", "m"),
        )
        # Example: a+4.0 -> ap4.0 ; a-4.0 -> am4.0

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
