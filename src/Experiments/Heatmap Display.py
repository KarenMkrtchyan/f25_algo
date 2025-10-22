import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re
import torch as t
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import Setup
from utils.model_config import load_model

model_name = "pythia-2.8b"
model = load_model(model_name)

text = "Which number is greater: 3 or 5?"
tokens = model.to_tokens(text)
tokens_str = model.to_str_tokens(tokens)
logits, cache = model.run_with_cache(tokens)

for i, tok in enumerate(tokens_str):
    print(i, repr(tok))

def is_relevant(tok: str) -> bool:
    t_clean = tok.lstrip('▁')  # remove leading space
    return bool(re.search(r'\d', t_clean)) or t_clean in [" greater", "less", ">", "<", "="]

relevant_tokens = [i for i, tok in enumerate(tokens_str) if is_relevant(tok)]
print("Relevant token indices:", relevant_tokens)

base_dir = "Attention_Heatmaps"
model_subfolder = os.path.join(base_dir, model_name)
os.makedirs(model_subfolder, exist_ok=True)

attn_sum = None

num_layers = model.cfg.n_layers
for layer in range(num_layers):
    key_name = f"blocks.{layer}.attn.hook_pattern"
    attn_layer = cache[key_name][0]
    attn_layer = t.softmax(attn_layer, dim=-1)

    attn_avg = t.mean(attn_layer, dim=0)
    attn_relevant_avg = attn_avg[relevant_tokens, :][:, relevant_tokens]

    if attn_sum is None:
        attn_sum = attn_avg
    else:
        attn_sum += attn_avg

    labels = [tokens_str[i] for i in relevant_tokens]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        attn_relevant_avg.detach().cpu().numpy(),
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.title(f"Layer {layer} Aggregated Attention (all heads)")
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")

    filename = os.path.join(model_subfolder, f"layer{layer}_aggregated.png")
    plt.savefig(filename)
    plt.close()

    

attn_model_avg = attn_sum / num_layers
attn_relevant_model = attn_model_avg[relevant_tokens, :][:, relevant_tokens]

plt.figure(figsize=(6, 5))
sns.heatmap(
    attn_relevant_model.detach().cpu().numpy(),
    xticklabels=labels,
    yticklabels=labels,
    cmap="viridis",
    annot=True,
    fmt=".2f"
)
plt.title(f"{model_name} - Aggregated Attention Across All Layers and Heads")
plt.xlabel("Key tokens")
plt.ylabel("Query tokens")

filename = os.path.join(model_subfolder, f"Aggregated_all_layers.png")
plt.savefig(filename)
plt.close()
