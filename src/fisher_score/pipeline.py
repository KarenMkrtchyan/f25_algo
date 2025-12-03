"""
Run this pipeline to look for the relevant MLP neurons for digit wise-comparions
"""

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from src.fisher_score.utils.fisherCalculations import FisherCalculations
from src.fisher_score.utils.fisherDataGen import FisherDataGenerator
from src.fisher_score.utils.globals import modelslist


drive_path = '/content/drive/My Drive/algo/fisher_scores/3_digit_llama/'
os.makedirs(drive_path, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
gen = FisherDataGenerator(seed=42)
groups = gen.fisher_score_groups()

for cfg in modelslist:
    if cfg['start_layer'] == -1:
        continue

    model = HookedTransformer.from_pretrained(cfg['name'], device='cuda')
    model.eval()

    model_fisher = FisherCalculations(model=model, data=groups, device=device, start_layer=cfg['start_layer'], end_layer=cfg['end_layer'])
    for digit in ["hundreds", "tens", "units"]:
        model_fisher.calc_in_class_stats(digit_position=digit)
        model_fisher.calc_global_mean()
        model_fisher.calc_between_class_var()
        fisher_scores = model_fisher.calc_fisher()

        n_layers = fisher_scores.shape[0]

        for layer_idx in range(n_layers):
            # Extract one layer’s fisher scores (shape [4096])
            layer_scores = fisher_scores[layer_idx]

            # Make it a DataFrame (each neuron’s score in one column)
            df = pd.DataFrame(layer_scores, columns=["fisher_score"])

            # Construct a unique file name per layer
            file_name = f"{cfg['start_layer'] + layer_idx}_{digit}.csv"
            full_path = os.path.join(drive_path, file_name)

            # Save to CSV
            df.to_csv(full_path, index=False)

            print(f"Saved {full_path}")
            torch.cuda.empty_cache()
