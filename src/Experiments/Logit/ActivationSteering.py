import os
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, build_numeric_batches, plot_activation_steering
import transformer_lens.utils as utils

dataset = build_dataset(n=100, low=1000, high=9999)

#model_name = "pythia-70m"
model_name = "qwen2.5-3b"
model = load_model(model_name)
model.set_use_attn_result(True)
device = get_device()

yes_id = model.to_single_token(" Yes")
no_id = model.to_single_token(" No")

Attention_Layer = 24
Attention_Head = 7

batches_base, batches_src, batches_ans = build_numeric_batches(model, dataset, yes_id, no_id, device)
num_batches = len(batches_src)
print(f"Batched into {num_batches} batches")
print("\n")

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "4digit")
output_folder = os.path.join(digit_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
plot_path = os.path.join(output_folder, f"Activation_Steering_Layer{Attention_Layer}_Head{Attention_Head}.png")

pc1 = plot_activation_steering(
    model=model,
    batches_base=batches_base,
    batches_src=batches_src,
    yes_id=yes_id,
    no_id=no_id,
    layer=Attention_Layer,
    head=Attention_Head,
    alpha=20.0,
    save_path = plot_path,
    device=device,
)
