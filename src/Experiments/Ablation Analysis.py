import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup
from utils.Setup import t

from utils.model_config import load_model
from Interpretability import get_ablation_scores

model = load_model("pythia-70m")
text = "Which number is greater: 3 or 5?"
tokens = model.to_tokens(text)
ablation_scores = get_ablation_scores(model, tokens)

summary_stats = {
    "layer_mean": t.mean(ablation_scores, dim=1),
    "layer_min": t.min(ablation_scores, dim=1),
    "layer_max": t.max(ablation_scores, dim=1),
    "head_mean": t.mean(ablation_scores, dim=0),
    "head_min": t.min(ablation_scores, dim=0),
    "head_max": t.max(ablation_scores, dim=0),
    "overall_mean": t.mean(ablation_scores).item(),
    "overall_min": t.min(ablation_scores).item(),
    "overall_max": t.max(ablation_scores).item(),
}
print(summary_stats)