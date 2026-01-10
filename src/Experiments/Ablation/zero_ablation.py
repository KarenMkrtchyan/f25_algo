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

model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#%%

task = "greater_than_4_digit"


#%%

forward_hooks = []

pos_list = [13,14,15]

for ablate_pos in pos_list:

    head_list=[
        {
            "layer": 28,
            "head": 10,
            "pos": ablate_pos,
        },
    ]

    for head in head_list:
        hook_name = utils.get_act_name("z", head["layer"])

        forward_hooks.append(
            (
                hook_name,
                partial(head_zero_ablation_hook_by_pos,
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
        df = run_benchmark_logits(
            model=model,
            task_name=task,
            num_fewshot=0,
            limit=1000,
            run=1,
            ablated_head=ablated_head_str,
            ablated_pos=head_list[0]["pos"],
        )
    # %%
