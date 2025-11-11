#!/usr/bin/env python3
"""
Plot layer-wise and per-head attention patterns for multiple models.

Compatible with the output of Attention_Patterns.py
(works for Qwen2.5-{3B,4B,7B} and Pythia models).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ────────────────────────────────────────────────
# Per-layer summary plot
# ────────────────────────────────────────────────

def plot_layer_summary(df, model_name, output_dir):
    """Plots median, max, and min mean attention per layer."""
    summary = (
        df.groupby("layer")
        .agg(
            median_attention=("mean_attention_toward", "median"),
            max_attention=("mean_attention_toward", "max"),
            min_attention=("mean_attention_toward", "min"),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x="layer", y="median_attention", label="Median", linewidth=2)
    sns.lineplot(data=summary, x="layer", y="max_attention", label="Max", linestyle="--", linewidth=1.5)
    sns.lineplot(data=summary, x="layer", y="min_attention", label="Min", linestyle=":", linewidth=1.5)

    plt.title(f"Layer-wise Attention Summary ({model_name})", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Attention Toward Target Tokens", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{model_name}_layer_summary.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {out_path}")


# ────────────────────────────────────────────────
# Per-head attention plot
# ────────────────────────────────────────────────

def plot_per_head(df, model_name, output_dir):
    """Plots per-head median attention per layer."""
    per_head = (
        df.groupby(["layer", "head"])
        .agg(median_attention=("mean_attention_toward", "median"))
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=per_head, x="layer", y="median_attention", hue="head", palette="tab10")
    plt.title(f"Per-Head Attention Across Layers ({model_name})", fontsize=13)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Median Mean Attention Toward Target Tokens", fontsize=12)
    plt.legend(title="Head", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{model_name}_per_head.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# ────────────────────────────────────────────────
# CSV Processing Function
# ────────────────────────────────────────────────

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"\n Processing {os.path.basename(csv_path)}")
    print(f"Columns: {df.columns.tolist()}")

    if "mean_attention_toward" not in df.columns:
        raise ValueError("Expected 'mean_attention_toward' column not found.")

    # Normalize model name for pretty titles
    base = os.path.basename(csv_path)
    model_name = base.split("--")[0]
    model_pretty = (
        model_name.replace("pythia-", "Pythia ")
        .replace("qwen2.5-", "Qwen2.5 ")
    )

    # Create output dir for plots
    output_dir = os.path.join(os.path.dirname(csv_path), "Plots_Attention")
    os.makedirs(output_dir, exist_ok=True)

    # Make both plots
    plot_layer_summary(df, model_pretty, output_dir)
    plot_per_head(df, model_pretty, output_dir)


# ────────────────────────────────────────────────
# Main Entry
# ────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = "/Users/sydverma/Documents/f25_algo/Results/Concatenated_Attention_Patterns"
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

    for csv_name in csv_files:
        csv_path = os.path.join(base_dir, csv_name)
        try:
            process_csv(csv_path)
        except Exception as e:
            print(f" Skipped {csv_name}: {e}")

    print("\n All attention plots generated successfully!")
