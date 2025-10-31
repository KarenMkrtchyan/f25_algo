import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import concatenated_ablation_patterns, get_top_ablated_heads, attach_head_ablation_hooks

model_name = "pythia-70m"
yaml_path = "tasks/greater_than/greater_than.yaml"
n_examples = 5
n_shots = 2

concat_df, dfs = concatenated_ablation_patterns(
    model_name=model_name,
    yaml_path=yaml_path,
    n_examples=n_examples,
    n_shots=n_shots
)

task = "greater_than"

results_folder = "Results"
output_folder = os.path.join(results_folder, "Concatenated_Ablated_Heads")
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, f"{model_name}--{task}--{n_examples}examples--{n_shots}shots--concat_df.csv")
concat_df.to_csv(csv_path, index=False)

group_by = "sum"
n = 10
top_heads_df, layers_list, heads_list, ablation_scores = get_top_ablated_heads(concat_df = concat_df, group_by = "sum")
print(top_heads_df)
print(layers_list)
print(heads_list)
print(ablation_scores)

csv_path = os.path.join(output_folder, f"{model_name}--{task}--{n_examples}examples--{n_shots}shots--top_{n}_ablated_heads_df.csv")
top_heads_df.to_csv(csv_path, index=False)

ablated_model = attach_head_ablation_hooks(model_name = model_name, layers = layers_list, heads = heads_list)