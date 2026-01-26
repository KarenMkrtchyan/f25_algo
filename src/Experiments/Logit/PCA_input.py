import os
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, build_numeric_batches, plot_head_input_PCA
import transformer_lens.utils as utils

dataset = build_dataset(n=100, low=1000, high=9999)

#model_name = "pythia-70m"
#model_name = "qwen2.5-3b"
#model_name = "qwen3-1.7b"
#model_name = "phi-3"
#model_name = "gemma-2-9b-it"
model_name = "llama3-8b-it"

model = load_model(model_name, torch_dtype=t.bfloat16)
model.set_use_attn_result(True)
device = get_device()

yes_id = model.to_single_token(" Yes")
no_id = model.to_single_token(" No")
#yes_id = model.to_single_token("Yes")
#no_id = model.to_single_token("No")

#Attention_Layer = 15
layers = [18, 19, 20, 21, 22, 23]

batches_base, batches_src, batches_ans = build_numeric_batches(model, dataset, yes_id, no_id, device)
num_batches = len(batches_src)
print(f"Batched into {num_batches} batches")
print("\n")

for Attention_Layer in layers:

    print(f"Computing PCA on input of Layer {Attention_Layer}")
    results_folder = "Results"
    display_folder = os.path.join(results_folder, "Digit_Experiment")
    digit_folder = os.path.join(display_folder, "4digit")
    output_folder = os.path.join(digit_folder, f"{model_name}")
    os.makedirs(output_folder, exist_ok=True)
    PCA_path = os.path.join(output_folder, f"PCA_input_Layer{Attention_Layer}.png")
    csv_path = os.path.join(output_folder, f"PCA_input_Layer{Attention_Layer}.csv")

    df = plot_head_input_PCA(
        model,
        batches_base,
        batches_src,
        layer = Attention_Layer,
        save_path = PCA_path
    )

    df.to_csv(csv_path, index = False)
    print("Finished PCA on input")
    print("\n")
