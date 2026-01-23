# src/steering/plot_qwen_fliprate.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="output/qwen_fliprate.csv")
    ap.add_argument("--out_dir", type=str, default="output/plots")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # ---------------------------
    # 1) Flip rate + accuracies
    # ---------------------------
    flip_rate = df["flipped"].mean()
    base_acc = df["base_correct"].mean()
    steered_acc = df["steered_correct"].mean()

    plt.figure()
    plt.bar(["flip_rate", "base_acc", "steered_acc"], [flip_rate, base_acc, steered_acc])
    plt.title("Flip rate and accuracy")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/rates.png")
    plt.close()

    # ---------------------------
    # 2) Histogram of margin shifts
    # ---------------------------
    plt.figure()
    plt.hist(df["margin_shift"], bins=50)
    plt.title("Distribution of margin shift (steered - base)")
    plt.xlabel("margin_shift (logp(Yes)-logp(No)) change")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/margin_shift_hist.png")
    plt.close()

    # ---------------------------
    # 3) Scatter: base margin vs steered margin
    # ---------------------------
    plt.figure()
    plt.scatter(df["base_margin"], df["steered_margin"], s=8, alpha=0.5)
    plt.title("Base vs steered margin")
    plt.xlabel("base_margin = logp(Yes)-logp(No)")
    plt.ylabel("steered_margin = logp(Yes)-logp(No)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/base_vs_steered_margin.png")
    plt.close()

    # ---------------------------
    # 4) How many flips by label type
    # ---------------------------
    flip_by_label = df.groupby("label")["flipped"].mean()
    plt.figure()
    plt.bar(flip_by_label.index.tolist(), flip_by_label.values.tolist())
    plt.title("Flip rate by ground-truth label")
    plt.ylabel("flip_rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/flip_by_label.png")
    plt.close()

    # Print quick text summary too
    print("Saved plots to:", args.out_dir)
    print("flip_rate:", float(flip_rate))
    print("base_acc:", float(base_acc))
    print("steered_acc:", float(steered_acc))


if __name__ == "__main__":
    main()
