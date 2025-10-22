import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup
from utils.Setup import t

from utils.model_config import load_model
from Interpretability import logit_attribution

model = load_model("pythia-70m")
text = "Which number is greater: 3 or 5?"
tokens = model.to_tokens(text)
logits, cache = model.run_with_cache(tokens)

l1_results = cache["z", 0]
l2_results = cache["z", 1]
embed = cache["hook_embed", 0]
tokens = tokens.squeeze(0)

attribution_tensor = logit_attribution(model, embed, l1_results, l2_results, tokens)

direct = attribution_tensor[:, 0]
l1 = attribution_tensor[:, 1:]
l2 = attribution_tensor[:, 1:]

logit_summary = {
    "direct_mean": t.mean(direct),
    "direct_min": t.min(direct),
    "direct_max": t.max(direct),
    "l1_mean": t.mean(l1),
    "l1_min": t.min(l1),
    "l1_max": t.max(l1),
    "l2_mean": t.mean(l2),
    "l2_min": t.min(l2),
    "l2_max": t.max(l2),
}
print(logit_summary)