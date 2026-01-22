import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, build_dataset_space, build_numeric_batches, compute_baselines, numeric_metric, full_dla_pipeline_all_positions, full_dla_pipeline_normalized
import transformer_lens.utils as utils

dataset = build_dataset(n=100, low=1000, high=9999)
#dataset = build_dataset_space(n=100, low=1000, high=9999)

model_name = "pythia-70m"
#model_name = "qwen2.5-3b"
#model_name = "qwen3-1.7b"
#model_name = "phi-3"
model = load_model(model_name)
device = get_device()

yes_id = model.to_single_token(" Yes")
no_id = model.to_single_token(" No")
#yes_id = model.to_single_token("Yes")
#no_id = model.to_single_token("No")

batches_base, batches_src, batches_ans = build_numeric_batches(model, dataset, yes_id, no_id, device)
num_batches = len(batches_src)
print(f"Batched into {num_batches} batches")
print("\n")

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "4digit")
output_folder = os.path.join(digit_folder, f"{model_name}")
dla_folder = os.path.join(output_folder, "DLA")
os.makedirs(dla_folder, exist_ok=True)

full_dla_pipeline_normalized(model, batches_base, batches_src, yes_id, no_id, output_folder = dla_folder)
