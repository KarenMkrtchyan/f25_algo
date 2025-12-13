#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Interpretability import head_zero_ablation_hook_by_pos
import torch
from test_suite.eval_logits import run_benchmark_logits

from functools import partial
from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(False)
torch.cuda.empty_cache()

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-3b",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    )

#%%
example_prompt = "Is 7912 > 1510? Answer:"
example_answer = "NO"

utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)
#%%

per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)

# 2. Project onto your Yes-No direction
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)

# 3. Plot
imshow(
    per_head_logit_diffs,
    labels={"x": "Head", "y": "Layer"},
    title="Logit Difference From Each Head",
    width=600,
)

# %%


