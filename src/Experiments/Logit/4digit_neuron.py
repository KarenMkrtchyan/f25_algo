import os
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, build_numeric_batches, compute_baselines, patch_mlp_neurons, save_sorted_neuron_importance, plot_neuron_scores, numeric_metric
from neel_plotly import imshow
import transformer_lens.utils as utils

dataset = build_dataset(n=100, low=1000, high=9999)

#model_name = "pythia-70m"
model_name = "qwen2.5-3b"
model = load_model(model_name)
device = get_device()
layer = 23

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

scores = patch_mlp_neurons(
    model=model,
    layer=layer,
    batches_base=batches_base,
    batches_src=batches_src,
    numeric_metric=numeric_metric,
    CLEAN_BASELINE=CLEAN_BASELINE,
    CORRUPTED_BASELINE=CORRUPTED_BASELINE,
    yes_id=yes_id,
    no_id=no_id
)

print("Neuron patching complete")
print("\n")

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "4digit")
output_folder = os.path.join(digit_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, f"Layer{layer}_neurons.csv")
plot_path = os.path.join(output_folder, f"Layer{layer}_neurons.png")

df = save_sorted_neuron_importance(scores, layer, csv_path)
plot_neuron_scores(scores, layer, plot_path)