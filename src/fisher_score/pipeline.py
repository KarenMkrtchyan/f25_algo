"""
Run this pipeline to look for the relevant MLP neurons for digit wise-comparions
"""

import torch
from transformer_lens import HookedTransformer
from src.fisher_score.utils.fisherDataGen import FisherDataGenerator
from src.fisher_score.utils.fisherCalculations import FisherCalculations
from src.fisher_score.utils.modelList import modelslist

device = "cuda" if torch.cuda.is_available() else "cpu"
gen = FisherDataGenerator(seed=42)
groups = gen.fisher_score_groups()

for cfg in modelslist:
    if cfg['start_layer'] == -1:
        continue

    model = HookedTransformer.from_pretrained(cfg['name'], device=device)
    model.eval()

    # TODO: Run the model once with the prompts, catche the activations, and run the fisher calculations on that

    model_fisher = FisherCalculations(model=model, data=groups, device=device, layer=cfg['start_layer'])
    for digit in ["hundreds", "tens", "units"]:
        model_fisher.calc_in_class_stats_single_pass(digit_position='units')
        model_fisher.calc_global_mean()
        model_fisher.calc_between_class_var()
        fisher_scores = model_fisher.calc_fisher()
    # Example: get top-10 neurons for this layer
    # topk_vals, topk_idx = torch.topk(fisher_scores, k=10)
    # print(f"Top neurons (layer={cfg['start_layer']}):", list(zip(topk_idx.tolist(), topk_vals.tolist())))

# Scale this across layers of the mode
# Viz the data
# look for patterns and 0 out some neurons to see effect on classfication
# repeat to multiple models
