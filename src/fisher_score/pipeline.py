"""
import ...

gen data using util

load in configs from yaml

calc fisher score for each layer and append to file

"""

import torch
from transformer_lens import HookedTransformer
from src.fisher_score.utils.fisherDataGen import FisherDataGenerator
from src.fisher_score.utils.fisherCalculations import FisherCalculations

device = "cuda" if torch.cuda.is_available() else "cpu"
# TODO: sweep through models by loading them from a yaml
model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model.eval()


gen = FisherDataGenerator(seed=42)
groups = gen.fisher_score_groups()


model_fisher = FisherCalculations(model=model, data=groups, device=device)
model_fisher.calc_in_class_stats(digit_position='units')