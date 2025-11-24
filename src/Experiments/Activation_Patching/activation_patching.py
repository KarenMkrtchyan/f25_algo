#%%
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import concatenated_attention_patterns, get_top_attention_heads, attach_head_ablation_hooks
import torch
from test_suite.eval import run_benchmark

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


torch.cuda.empty_cache()

#%%
from transformer_lens.patching import get_act_patch_attn_head_out_by_pos
from transformer_lens.HookedTransformer import HookedTransformer


model = HookedTransformer.from_pretrained("qwen2.5-3b")
clean_prompt = "Is 9432 > 8231? Answer: "
corrupted_prompt = "Is 8231 > 9432? Answer: "

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

clean_ans_token = model.to_single_token("Yes")
corrupted_ans_token = model.to_single_token("No")

_, clean_cache = model.run_with_cache(clean_tokens)

def logit_diff_metric(logits):
    last_token_logits = logits[0, -1, :]

    clean_logit = last_token_logits[clean_ans_token]
    corrupted_logit = last_token_logits[corrupted_ans_token]

    return clean_logit - corrupted_logit

# returns the tensor of the patching metric for 
# each patch. Has shape [n_layers, pos, n_heads]

#%%
tensor_result = get_act_patch_attn_head_out_by_pos(
    model=model,
    corrupted_tokens=corrupted_tokens,
    clean_cache=clean_cache,
    patching_metric=logit_diff_metric
)

torch.save(tensor_result, 'results/patching_result.pt')
#%%
#tensor_result = torch.load(f'results/patching_result.pt')

# %%

# plot one heatmap per position

n_positions = tensor_result.shape[1]

fig = make_subplots(
    rows=1, cols=n_positions,
    subplot_titles=[f"Pos {i}" for i in range(n_positions)],
    horizontal_spacing=0.02
)

for pos in range(n_positions):
    heatmap_data = tensor_result[:, pos, :].detach().cpu()
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            colorscale='RdBu_r',
            zmid=0,
            showscale=(pos == n_positions - 1)
        ),
        row=1, col=pos+1
    )

fig.update_layout(
    height=400,
    width=200 * n_positions
)

fig.write_image(f'results/heatmap_by_position4.png', width=200*n_positions, height=400)
fig.show()

#%%
tokens = model.to_str_tokens(clean_tokens)

with open(f'results/tokens4.txt', 'w') as f:
    f.write(str(tokens) + "\n")

# %%
