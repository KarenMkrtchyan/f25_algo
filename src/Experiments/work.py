import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model

import torch as t
import pandas as pd
import re
from transformer_lens import HookedTransformer

def attention_pattern_toward_each_token(model_name, text):
    model = load_model(model_name)

    tokens = model.to_tokens(text)
    tokens_str = model.to_str_tokens(tokens)
    logits, cache = model.run_with_cache(tokens)

    print("Token indices:")
    for i, tok in enumerate(tokens_str):
        print(i, repr(tok))

    def is_relevant(tok: str) -> bool:
        t_clean = tok.lstrip('▁')
        return bool(re.search(r'\d', t_clean)) or t_clean in [" greater", "less", ">", "<", "="]

    relevant_indices = [i for i, tok in enumerate(tokens_str) if is_relevant(tok)]
    print("Relevant token indices:", relevant_indices)

    all_stats = []

    for layer in range(model.cfg.n_layers):
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]

        for head in range(model.cfg.n_heads):
            attn_head = attn[head]

            for target_idx in relevant_indices:
                mean_to_token = attn_head[:, target_idx].mean().item()
                max_to_token = attn_head[:, target_idx].max().item()

                all_stats.append({
                    "layer": layer,
                    "head": head,
                    "target_index": target_idx,
                    "target_token": tokens_str[target_idx],
                    "mean_attention_toward": mean_to_token,
                    "max_attention_toward": max_to_token,
                })

    df = pd.DataFrame(all_stats)
    return df

model_name = "pythia-70m"

df = attention_pattern_toward_each_token("pythia-70m", "which number is greater 5 or 3")
df.to_csv("attention_weights.csv", index=False)
print(df)

agg = df.groupby(["layer", "head"])["mean_attention_toward"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=agg, x="layer", y="mean_attention_toward", hue="head", palette="tab10")
plt.title("Change in Attention Across Layers (per Head)")
plt.xlabel("Layer")
plt.ylabel("Mean Attention to Relevant Tokens")
plt.legend(title="Head", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig(f"attention_change_per_head--{model_name}.png", dpi=300, bbox_inches="tight")
plt.show()