import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch as t
import pandas as pd
from utils import Setup
from utils.model_config import load_model


model_name = "qwen2-7b"
model = load_model(model_name)

text = "Which number is greater: 3 or 5?"
tokens = model.to_tokens(text)
tokens_str = model.to_str_tokens(tokens)

logits, cache = model.run_with_cache(tokens)

tokens_str = model.to_str_tokens(tokens)
for i, tok in enumerate(tokens_str):
    print(i, repr(tok))

def is_relevant(tok: str) -> bool:
    t_clean = tok.lstrip('▁')  # remove leading space
    return bool(re.search(r'\d', t_clean)) or t_clean in [" greater", "less", ">", "<", "="]

relevant_tokens = [i for i, tok in enumerate(tokens_str) if is_relevant(tok)]
print("Relevant token indices:", relevant_tokens)


all_stats = []
for layer in range(model.cfg.n_layers):
    key_name = f"blocks.{layer}.attn.hook_pattern"
    attn = cache[key_name][0]

    for head in range(model.cfg.n_heads):
        attn_head = attn[head]
        if len(relevant_tokens) > 0:
            attn_relevant = attn_head[relevant_tokens, :][:, relevant_tokens]
            mean_val = t.mean(attn_relevant).item()
            min_val = t.min(attn_relevant).item()
            max_val = t.max(attn_relevant).item()
        else:
            mean_val = min_val = max_val = None

        all_stats.append({
            "layer": layer,
            "head": head,
            "mean": mean_val,
            "min": min_val,
            "max": max_val
        })

results_folder = "Results"
output_folder = os.path.join(results_folder, "Attention Patterns")
os.makedirs(output_folder, exist_ok=True)

df_stats = pd.DataFrame(all_stats)
print(df_stats)

csv_path = os.path.join(output_folder, f"{model_name}_attention_pattern.csv")
df_stats.to_csv(csv_path, index=False)
