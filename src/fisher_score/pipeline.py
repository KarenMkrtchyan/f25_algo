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
from src.fisher_score.utils.modelList import modelslist


drive_path = '/content/drive/My Drive/algo/fisher_scores/'
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
        df = pd.DataFrame(fisher_scores)

        file_name = f"fisher_{cfg['start_layer']}_{digit}_{cfg['name'][0:3]}.csv"
        full_path = drive_path + file_name

        df.to_csv(full_path, index=False)
    del model
    torch.cuda.empty_cache()

# Data viz 
data_path = drive_path + "data/"
for file in os.listdir(data_path):
  full_path = os.path.join(data_path, file)
  try:
    df = pd.read_csv(full_path)

    df.hist(bins=40)
    plt.title(f'Distribution of Fisher Scores for {file}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    # Save instead of showing
    fp = drive_path + 'plots/' + file[0:-3]

    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()
  except:
    print("not valid file")
