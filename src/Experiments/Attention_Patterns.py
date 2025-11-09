import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_summary(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}")
    print(f"Columns: {df.columns.tolist()}")

    # Check for new-style attention columns
    if "mean_attention_toward" not in df.columns:
        raise ValueError(f"Unexpected CSV format for {csv_path}")

    # Get clean model name (e.g. 'pythia-410m' or 'qwen2.5-3b')
    base = os.path.basename(csv_path)
    model_name = base.split("--")[0]
    model_short = model_name.replace("pythia-", "Pythia ").replace("qwen2.5-", "Qwen2.5 ")

    # --- Compute per-layer statistics ---
    summary = (
        df.groupby("layer")
        .agg(
            median_attention=("mean_attention_toward", "median"),
            max_attention=("mean_attention_toward", "max"),
            min_attention=("mean_attention_toward", "min")
        )
        .reset_index()
    )

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x="layer", y="median_attention", label="Median", linewidth=2)
    sns.lineplot(data=summary, x="layer", y="max_attention", label="Max", linestyle="--", linewidth=1.5)
    sns.lineplot(data=summary, x="layer", y="min_attention", label="Min", linestyle=":", linewidth=1.5)

    plt.title(f"Layer-wise Attention Summary ({model_short})", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Attention Toward Target Tokens", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # --- Save as unique image per model ---
    output_dir = os.path.join(os.path.dirname(csv_path), "Plots_Attention")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_layer_summary.png")

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    base_dir = "/Users/sydverma/Documents/f25_algo/Results/Concatenated_Attention_Patterns"

    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

    for csv_name in csv_files:
        csv_path = os.path.join(base_dir, csv_name)
        print(f"\n📂 Processing {csv_name} ...")
        try:
            plot_attention_summary(csv_path)
        except Exception as e:
            print(f"Skipped {csv_name}: {e}")

