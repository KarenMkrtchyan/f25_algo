"""
Run this pipeline to look for the relevant MLP neurons for digit wise-comparions
"""

import torch
from transformer_lens import HookedTransformer
from src.fisher_score.utils.fisherDataGen import FisherDataGenerator
from src.fisher_score.utils.fisherCalculations import FisherCalculations
from src.fisher_score.utils.modelList import modelslist

device = "cuda" if torch.cuda.is_available() else "cpu"

for model in modelslist:
    if model.start_layer == -1:
        continue

    model = HookedTransformer.from_pretrained(model.name)
    model.eval()


    # gen = FisherDataGenerator(seed=42)
    # groups = gen.fisher_score_groups()


    # model_fisher = FisherCalculations(model=model, data=groups, device=device)
    # model_fisher.calc_in_class_stats(digit_position='units')

# Scale this across layers of the mode
# Viz the data
# look for patterns and 0 out some neurons to see effect on classfication
# repeat to multiple models