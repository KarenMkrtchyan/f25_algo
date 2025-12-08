#%%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Interpretability import head_mean_ablation_hook_by_pos
import torch
from test_suite.eval import run_benchmark

from functools import partial
from transformer_lens import HookedTransformer, utils
from data.head_list import head_list_qwen17

torch.set_grad_enabled(False)
torch.cuda.empty_cache()

model = HookedTransformer.from_pretrained("Qwen/Qwen3-1.7b")

tasks = ["greater_than_4_digit"]

for task in tasks:
    # # of token positions
    if (task == "greater_than_4_digit"): n_positions = 16

    for head in head_list_qwen17:
        hook_name = utils.get_act_name("z", head["layer"])
        results = []

        for pos in range(n_positions):
            forward_hooks = [
                (
                    hook_name,
                    partial(head_mean_ablation_hook_by_pos,
                            head_index_to_ablate=head["head"],
                            pos_to_ablate=pos,
                            ),
                )
            ]

            with model.hooks(fwd_hooks=forward_hooks):
                #ablated_logits = model(tokens, return_type="logits")
                df = run_benchmark(
                    model=model,
                    task_name=task,
                    num_fewshot=0,
                    limit=1000,
                    run=1,
                    ablated_head=f"L{head['layer']}H{head['head']}",
                    ablated_pos=pos,
                )


# %%
