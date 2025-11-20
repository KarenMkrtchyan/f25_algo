import os
import sys
import json
import torch as t
import numpy as np
import pandas as pd
import einops
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, ActivationCache
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from Interpretability import build_dataset, load_or_generate_parquet, run_patching, plot_attention_head_heatmap, plot_mlp_patch_bar, plot_resid_patch_bar
#from sklearn.decomposition import PCA

dataset = build_dataset(n=1500, low = 1000, high = 9999)

model_name = "pythia-70m"
model = load_model(model_name)
Yes_index = model.to_single_token(" Yes")
No_index = model.to_single_token(" No")

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiments")
output_folder = os.path.join(display_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "4digit.parquet")

df = load_or_generate_parquet(model, dataset, output_path, Yes_index, No_index)
patch_effects = run_patching(df, dataset, model)

output_path = os.path.join(output_folder, "attention_heatmap.png")
plot_attention_head_heatmap(patch_effects, model, output_path)
output_path = os.path.join(output_folder, "mlp_heatmap.png")
plot_mlp_patch_bar(patch_effects, model, output_path)
output_path = os.path.join(output_folder, "residual_layer_heatmap.png")
plot_resid_patch_bar(patch_effects, model, output_path)