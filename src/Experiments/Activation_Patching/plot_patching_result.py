#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from data.prompts import prompts
from transformer_lens import HookedTransformer

# %%

model = HookedTransformer.from_pretrained("qwen2.5-3b")

#%%

# plot one heatmap per position
patching_result = torch.load("results/patching_result016.pt")
prompt = prompts[16]["clean_prompt"]

#%%
tokens = model.to_str_tokens(prompt)
print(tokens)

#%%

n_positions = patching_result.shape[1]

fig = make_subplots(
    rows=1, cols=n_positions,
    subplot_titles=[f"Pos {i}" for i in range(n_positions)],
    horizontal_spacing=0.02
)

for pos in range(n_positions):
    heatmap_data = patching_result[:, pos, :].detach().cpu()
    
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

results_dir = "results"

fig.write_image(
    os.path.join(results_dir, "heatmap_by_position_last_prompt.png"),
    width=200 * n_positions,
    height=400,
)
fig.show()

#%%