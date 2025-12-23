#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Interpretability import head_mean_ablation_hook_by_pos
import torch
from test_suite.eval import run_benchmark

from functools import partial
from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(False)
torch.cuda.empty_cache()

model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3b")

#%%
task = "greater_than_4_digit"

head_list=[
    {
        "layer": 9,
        "head": 9,
        "pos": 5,
    },
]

forward_hooks = []

for head in head_list:
    hook_name = utils.get_act_name("z", head["layer"])

    forward_hooks.append(
        (
            hook_name,
            partial(head_mean_ablation_hook_by_pos,
                    head_index_to_ablate=head["head"],
                    pos_to_ablate=head["pos"],
                    ),
        )
    )

# build ablated head string
ablated_head_str=""
for head in head_list:
    ablated_head_str += f"L{head['layer']}H{head['head']}_"


with model.hooks(fwd_hooks=forward_hooks):
    #ablated_logits = model(tokens, return_type="logits")
    df = run_benchmark(
        model=model,
        task_name=task,
        num_fewshot=0,
        limit=1000,
        run=1,
        ablated_head=ablated_head_str,
        ablated_pos="11",
    )


# %%
