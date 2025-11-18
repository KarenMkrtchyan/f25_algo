import os
import sys
import torch
import numpy as np
import einops
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, ActivationCache
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from Interpretability import build_dataset, logit_diff
#from sklearn.decomposition import PCA

dataset = build_dataset(n=1500, low = 1000, high = 9999)

model = load_model("pythia-70m")

Yes_index = model.to_single_token(" Yes")
No_index = model.to_single_token(" No")

clean_caches = []
corrupt_caches = []
logit_diffs = []

for clean, corrupt, a, b, label in tqdm(dataset):
    clean_logits, clean_cache = model.run_with_cache(clean, remove_batch_dim=False)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt, remove_batch_dim=False)
    
    clean_caches.append(clean_cache)
    corrupt_caches.append(corrupt_cache)
    logit_diffs.append(logit_diff(corrupt_logits, Yes_index, No_index))

print(len(clean_caches))
print(len(corrupt_caches))
print(len(logit_diffs))

'''
num_layers = model.cfg.n_layers
patch_effect_heads = torch.zeros(num_layers, model.cfg.n_heads)
patch_effect_mlps = torch.zeros(num_layers)

for i, (clean, corrupt, a, b, label) in enumerate(tqdm(dataset)):
    pos = get_last_pos(corrupt)
    clean_cache = clean_caches[i]
    corrupt_cache = corrupt_caches[i]
    base_ld = baseline_ldiffs[i]
    
    # Patch attention heads
    for L in range(num_layers):
        hook_name = f"blocks.{L}.attn.hook_result"
        patched_ld = patch_component(model, corrupt, clean_cache, hook_name, pos)
        patch_effect_heads[L] += patched_ld - base_ld
    
    # Patch MLPs
    for L in range(num_layers):
        hook_name = f"blocks.{L}.mlp.hook_post"
        patched_ld = patch_component(model, corrupt, clean_cache, hook_name, pos)
        patch_effect_mlps[L] += patched_ld - base_ld

patch_effect_heads /= len(dataset)
patch_effect_mlps /= len(dataset)
'''