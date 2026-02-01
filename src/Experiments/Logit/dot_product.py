import os
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, build_numeric_batches, compute_baselines, plot_head_to_neuron_dot_products
import transformer_lens.utils as utils

#dataset = build_dataset(n=100, low=1000, high=9999)
dataset = build_dataset(n=100, low=1000, high=9999)

#model_name = "pythia-70m"
model_name = "qwen2.5-3b"
#model_name = "pythia-160m"
model = load_model(model_name)
model.set_use_attn_result(True)
device = get_device()

yes_id = model.to_single_token(" Yes")
no_id = model.to_single_token(" No")

batches_base, batches_src, batches_ans = build_numeric_batches(model, dataset, yes_id, no_id, device)
num_batches = len(batches_src)
print(f"Batched into {num_batches} batches")
print("\n")

Attention_L = 24
Attention_H = 7
Neuron_L = 31
Neuron_num = 8338

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "4digit")
model_folder = os.path.join(digit_folder, f"{model_name}")
output_folder = os.path.join(model_folder, "Dot_Product")
os.makedirs(output_folder, exist_ok=True)
plot_path = os.path.join(output_folder, f"A(Layer{Attention_L}Head{Attention_H})_N(Layer{Neuron_L}Number{Neuron_num})_dot_product.png")
csv_path = os.path.join(output_folder, f"A(Layer{Attention_L}Head{Attention_H})_N(Layer{Neuron_L}Number{Neuron_num})_dot_product.csv")

df = plot_head_to_neuron_dot_products(
    model,
    batches_base,
    batches_src,
    Attention_L,
    Attention_H,
    Neuron_L,
    Neuron_num,
    title = "",
    save_path = plot_path
)

df.to_csv(csv_path, index = False)
