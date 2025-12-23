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

# Generate full 5-digit dataset
gen = FisherDataGenerator(seed=42)
groups = gen.fisher_score_groups(n_per_class=10)

DIGIT_POSITIONS = [
    "ten_thousands",
    "thousands",
    "hundreds",
    "tens",
    "units"
]

for cfg in modelslist:
    if cfg['start_layer'] == -1:
        continue

    print(f"Loading model {cfg['name']}")
    model = HookedTransformer.from_pretrained(cfg['name'], device=device)
    model.eval()

    fisher_engine = FisherCalculations(
        model=model,
        data=groups,
        device=device,
        start_layer=cfg['start_layer'],
        end_layer=cfg['end_layer']
    )

    for digit in DIGIT_POSITIONS:
        print(f"\n=== Computing Fisher for {digit} ===")
        fisher_engine.calc_in_class_stats(digit_position=digit)
        fisher_engine.calc_global_mean()
        fisher_engine.calc_between_class_var()
        fisher_scores = fisher_engine.calc_fisher()

        n_layers = fisher_scores.shape[0]

        for i in range(n_layers):
            layer_scores = fisher_scores[i]
            df = pd.DataFrame(layer_scores, columns=["fisher_score"])

            layer_name = cfg['start_layer'] + i
            filename = f"{layer_name}_{digit}.csv"
            full_path = os.path.join(drive_path, filename)

            df.to_csv(full_path, index=False)
            print(f"Saved {full_path}")

            torch.cuda.empty_cache()
