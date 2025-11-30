#%%
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import plotly.express as px
from data.prompts import prompts

torch.cuda.empty_cache()

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

#%%
from transformer_lens.patching import get_act_patch_attn_head_out_by_pos
from transformer_lens.HookedTransformer import HookedTransformer


model = HookedTransformer.from_pretrained("qwen2.5-3b")

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

head_counts = torch.zeros(n_layers, n_heads, dtype=torch.long)

#%% 
for i, item in enumerate(prompts):
    print(f"prompt {i+1}/{len(prompts)}")

    clean_prompt = item["clean_prompt"]
    corrupted_prompt = item["corrupted_prompt"]
    clean_label = item["clean_label"]
    corrupted_label= item["corrupted_label"]

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    clean_ans_token = model.to_single_token(clean_label)
    corrupted_ans_token = model.to_single_token(corrupted_label)

    _, clean_cache = model.run_with_cache(clean_tokens)

    def logit_diff_metric(logits):
        last_token_logits = logits[0, -1, :]

        clean_logit = last_token_logits[clean_ans_token]
        corrupted_logit = last_token_logits[corrupted_ans_token]

        return clean_logit - corrupted_logit

    # returns the tensor of the patching metric for 
    # each patch. Has shape [n_layers, pos, n_heads]
    tensor_result = get_act_patch_attn_head_out_by_pos(
        model=model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        patching_metric=logit_diff_metric
    )
    
    torch.save(tensor_result.cpu(), os.path.join(results_dir, f"patching_result{i:03d}.pt"))

    std_threshold = 5.0
    n_layers, n_positions, n_heads = tensor_result.shape

    for pos in range(n_positions):
        pos_scores = tensor_result[:, pos, :]

        flat = pos_scores.reshape(-1)
        mean_score = flat.mean()
        std_score = flat.std()

        cutoff = mean_score + (std_threshold * std_score)
        significant_indices = torch.nonzero(flat > cutoff).squeeze()

        if significant_indices.numel() > 0:
            if significant_indices.dim() == 0:
                significant_indices = [significant_indices]
                
            for idx in significant_indices:
                layer = int(idx // n_heads)
                head = int(idx % n_heads)
                head_counts[layer, head] += 1


#%%
# save metadata

torch.save(head_counts, os.path.join(results_dir, "head_counts.pt"))
torch.save(
    {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_positions": n_positions,
        "num_prompts": len(prompts),
        "std_threshold": std_threshold,
    },
    os.path.join(results_dir, "run_metadata.pt"),
)

# plot heads heatmap

head_counts_cpu = head_counts.cpu()
flat_scores = head_counts_cpu.flatten().numpy()

n_layers, n_heads = head_counts_cpu.shape
labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]

series = pd.Series(flat_scores, index=labels)

top_n = 15
top_heads = series.sort_values(ascending=False).head(top_n)

fig_bar = px.bar(
    top_heads,
    x=top_heads.index,
    y=top_heads.values,
    title=f"Top {top_n} Most Important Attention Heads (Frequency Count)",
    labels={'x': 'Head ID (Layer/Head)', 'y': 'Total Frequency Count'},
    color=top_heads.values,
    color_continuous_scale="Viridis"
)

fig_bar.write_image(os.path.join(results_dir, "top_heads_bar.png"))
fig_bar.show()

# %%
