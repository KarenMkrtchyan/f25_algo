import os
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from Interpretability import build_dataset, compute_act_patching, get_logit_diff, paper_plot, build_numeric_batches, compute_baselines, numeric_metric, plot_all_patch_effects_paper
from neel_plotly import imshow
import transformer_lens.utils as utils

dataset = build_dataset(n=10, low=1000, high=9999)

model_name = "qwen2.5-3b"
model = load_model(model_name)
device = utils.get_device()

yes_id = model.to_single_token(" Yes")
no_id = model.to_single_token(" No")

batches_base, batches_src, batches_ans = build_numeric_batches(model, dataset, yes_id, no_id, device)
num_batches = len(batches_src)
print(f"Batched into {num_batches} batches")

CLEAN_BASELINE, CORRUPTED_BASELINE = compute_baselines(
    model, batches_base, batches_src, batches_ans
)
print("Baselines computed: ")
print("Clean:", CLEAN_BASELINE.item(), "Corrupted:", CORRUPTED_BASELINE.item())

patch_resid = compute_act_patching(
    model, numeric_metric, CLEAN_BASELINE, CORRUPTED_BASELINE, "resid_streams",
    batches_base, batches_src, batches_ans, num_batches
)
patch_heads = compute_act_patching(
    model, numeric_metric, CLEAN_BASELINE, CORRUPTED_BASELINE, "heads_last_pos",
    batches_base, batches_src, batches_ans, num_batches
)
patch_full = compute_act_patching(
    model, numeric_metric, CLEAN_BASELINE, CORRUPTED_BASELINE, "full",
    batches_base, batches_src, batches_ans, num_batches
)

patch_attn = patch_full[1]
patch_mlp  = patch_full[2]

print("Activation patching complete!")

tokens_str = model.to_str_tokens(model.to_tokens("Is 1234 > 5678? Answer:"))
results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit Experiment")
digit_folder = os.path.join(display_folder, "4digit")
output_folder = os.path.join(digit_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
out_path = os.path.join(output_folder, "numeric_full_patch.png")
plot_all_patch_effects_paper(
    model,
    patch_resid,
    patch_attn,
    patch_mlp,
    patch_heads,
    output_folder
)
