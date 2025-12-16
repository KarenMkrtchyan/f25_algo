import os
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, compute_act_patching, get_logit_diff, paper_plot, build_numeric_batches, compute_baselines, numeric_metric, plot_all_patch_effects_paper, save_sorted_head_importance, plot_component_scores
from neel_plotly import imshow
import transformer_lens.utils as utils

dataset = build_dataset(n=100, low=10000000, high=99999999)

#model_name = "pythia-70m"
model_name = "qwen2.5-3b"
model = load_model(model_name)
device = get_device()

yes_id = model.to_single_token(" Yes")
no_id = model.to_single_token(" No")

batches_base, batches_src, batches_ans = build_numeric_batches(model, dataset, yes_id, no_id, device)
num_batches = len(batches_src)
print(f"Batched into {num_batches} batches")
print("\n")

CLEAN_BASELINE, CORRUPTED_BASELINE = compute_baselines(
    model, batches_base, batches_src, yes_id, no_id
)
print("Baselines computed: ")
print("Clean:", CLEAN_BASELINE.item(), "Corrupted:", CORRUPTED_BASELINE.item())
print("\n")

patch_resid = compute_act_patching(
    model, numeric_metric, yes_id, no_id, CLEAN_BASELINE, CORRUPTED_BASELINE, "resid_streams",
    batches_base, batches_src, num_batches
)
patch_heads = compute_act_patching(
    model, numeric_metric, yes_id, no_id, CLEAN_BASELINE, CORRUPTED_BASELINE, "heads_last_pos",
    batches_base, batches_src, num_batches
)
patch_full = compute_act_patching(
    model, numeric_metric, yes_id, no_id, CLEAN_BASELINE, CORRUPTED_BASELINE, "full",
    batches_base, batches_src, num_batches
)

patch_attn = patch_full[1]
patch_mlp  = patch_full[2]
print(patch_full.shape)

print("Activation patching complete!")
print("\n")

tokens_str = model.to_str_tokens(model.to_tokens("Is 1234 > 5678? Answer:"))
results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "4digit")
output_folder = os.path.join(digit_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, "head_importance.csv")
component_plot_path = os.path.join(output_folder, "component_relevance.png")

save_sorted_head_importance(patch_heads, csv_path)
plot_all_patch_effects_paper(
    model,
    patch_resid,
    patch_attn,
    patch_mlp,
    patch_heads,
    output_folder
)
plot_component_scores(patch_full, model, component_plot_path)
