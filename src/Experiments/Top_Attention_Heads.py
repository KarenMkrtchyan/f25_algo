import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import concatenated_attention_patterns, get_top_attention_heads, attach_head_ablation_hooks

model_name = "qwen2.5-3b"
yaml_path = "tasks/greater_than/greater_than.yaml"
n_examples = 5
n_shots = 0

concat_df, dfs = concatenated_attention_patterns(
    model_name=model_name,
    yaml_path=yaml_path,
    n_examples=n_examples,
    n_shots=n_shots
)

task = "greater_than"

results_folder = "Results"
output_folder = os.path.join(results_folder, "Concatenated_Attention_Patterns")
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, f"{model_name}--{task}--{n_examples}examples--{n_shots}shots--concat_df.csv")
concat_df.to_csv(csv_path, index=False)

group_by = "sum"
sort_by = "mean_attention_toward"
n = 10
top_heads_df, layers_list, heads_list, attention_scores = get_top_attention_heads(concat_df = concat_df, sort_by = sort_by, group_by = "sum")
print(top_heads_df)
print(layers_list)
print(heads_list)
print(attention_scores)

csv_path = os.path.join(output_folder, f"{model_name}--{task}--{n_examples}examples--{n_shots}shots--top_{n}_attention_heads_df.csv")
top_heads_df.to_csv(csv_path, index=False)

agg = concat_df.groupby(["layer", "head"])["mean_attention_toward"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=agg, x="layer", y="mean_attention_toward", hue="head", palette="tab10")
plt.title("Change in Attention Across Layers (per Head)")
plt.xlabel("Layer")
plt.ylabel("Mean Attention to Relevant Tokens")
plt.legend(title="Head", bbox_to_anchor=(1.05, 1), loc="upper left")

plot_path = os.path.join(
    output_folder,
    f"{model_name}--{task}--{n_examples}examples--{n_shots}shots--top_{n}_attention_heads_plot.png"
)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

ablated_model = attach_head_ablation_hooks(model_name = model_name, layers = layers_list, heads = heads_list)
