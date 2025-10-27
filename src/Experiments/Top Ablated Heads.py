import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import run_model_with_ablation_analysis

def get_top_heads(model_name, text, threshold=0.1, n=5, layers_to_ablate=None, heads_to_ablate=None):
    """
    Run ablation analysis and return the top n most important heads.

    Args:
        model: HookedTransformer model
        text (str): Prompt text
        threshold (float): Minimum importance value to consider a head
        n (int): Number of top heads to return
        layers_to_ablate (list[int], optional): layers to ablate
        heads_to_ablate (list[int], optional): heads to ablate
    Returns:
        pd.DataFrame: Top heads with columns ["layer", "head", "importance"]
    """

    model = load_model(model_name)
    results = run_model_with_ablation_analysis(model, text, threshold = threshold)
    results_df = pd.DataFrame(results["important_heads"], columns=["layer", "head", "importance"])

    filtered_df = results_df.sort_values("importance", ascending=False)
    top_heads = filtered_df.head(n)
    
    layers_list = top_heads["layer"].tolist()
    heads_list = top_heads["head"].tolist()

    return layers_list, heads_list

x, y, = get_top_heads('pythia-70m', 'hello world')
print(x)
print(y)