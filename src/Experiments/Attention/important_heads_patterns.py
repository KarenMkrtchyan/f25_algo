#%%
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import plotly.express as px
from Interpretability.Functions import display_attention_heads

from transformer_lens.HookedTransformer import HookedTransformer
from IPython.display import display

model = HookedTransformer.from_pretrained("qwen2.5-3b")

#%%
text = "Is 9432 > 8231? Answer: "
_, cache = model.run_with_cache(text)

# We want to know what the LAST token (position -1) is looking at
last_pos = len(model.to_tokens(text)[0]) - 1

# "Who is L24H5 listening to?"
#%%
vis = display_attention_heads(model, text, cache, layer=0, position=last_pos)

with open("results/attention_heads_L0.html", "w") as f:
    f.write(str(vis))

vis

# %%
