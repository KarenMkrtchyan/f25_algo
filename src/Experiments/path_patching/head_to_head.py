#%%
import sys
import os
from data.prompts import prompts as prompt_list
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Interpretability.path_patching import path_patch, Node, IterNode, imshow, hist, path_patch_sender_to_receiver_logits
from transformer_lens.HookedTransformer import HookedTransformer
import torch as t
from torch import Tensor
from jaxtyping import Float, Int, Bool

# %%
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
t.set_grad_enabled(False)


model = HookedTransformer.from_pretrained(
    "qwen2.5-3b",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)

model.set_use_split_qkv_input(True)

#%%
prompts = [p["clean_prompt"] for p in prompt_list]
labels = [p["clean_label"] for p in prompt_list]

# Define the answers for each prompt, in the form (correct, incorrect)
answers = [("Yes", "No") if label == "Yes" else ("No", "Yes") for label in labels]

# Define the answer tokens (same shape as the answers)
yes_id = model.to_single_token("Yes")
no_id  = model.to_single_token("No")

answer_tokens = []
for label in labels:
    if label == "Yes":
        answer_tokens.append([yes_id, no_id])
    else:
        answer_tokens.append([no_id, yes_id])
answer_tokens = t.tensor(answer_tokens, device=device)

#%%
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)

    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

clean_tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
flipped_indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(clean_tokens))]
flipped_tokens = clean_tokens[flipped_indices]

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
flipped_logits, flipped_cache = model.run_with_cache(flipped_tokens)

clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
flipped_logit_diff = logits_to_ave_logit_diff(flipped_logits, answer_tokens)

print(
    "Clean string 0: " , model.to_string(clean_tokens[0]), "\n"
    "Flipped string 0: ", model.to_string(flipped_tokens[0])
)

print(f"Clean logit diff: {clean_logit_diff:.4f}")
print(f"Flipped logit diff: {flipped_logit_diff:.4f}")
#%%

def greater_than_metric_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    flipped_logit_diff: float = flipped_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
)-> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff - flipped_logit_diff)).item()

labels =[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]


#%%

# Patching from attention head -> final residual stream value

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=flipped_tokens,
    sender_nodes= IterNode('z'),
    receiver_nodes=Node("resid_post", 35),
    patching_metric=greater_than_metric_noising,
    verbose=True,
)

results

# %%

imshow(
    results['z'],
    title="Direct effect on logit diff (patch from head output -> final resid)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    border=True,
    width=600,
    margin={"r": 100, "l": 100},
    zmin=0.80, zmax=1.00
)
#%%

# Patching from residual stream-> final residual stream value (for each sequence position)

results = path_patch(
    model,
    orig_input=flipped_tokens,
    new_input=clean_tokens,
    sender_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    receiver_nodes=Node("resid_post", 35),
    patching_metric=greater_than_metric_noising,
    direct_includes_mlps=False, # gives similar results to direct_includes_mlps=True
    verbose=True,
)

#%%

# We get a dictionary where each key is a node name, and each value is a tensor of (layer, seq_pos)
assert list(results.keys()) == ['resid_pre', 'attn_out', 'mlp_out']

results_stacked = t.stack([
    results.T for results in results.values()
])

imshow(
    results_stacked,
    facet_col=0,
    facet_labels=['resid_pre', 'attn_out', 'mlp_out'],
    title="Results of denoising patching at residual stream",
    labels={"x": "Sequence position", "y": "Layer", "color": "Logit diff variation"},
    x=labels,
    xaxis_tickangle=45,
    width=1300,
    margin={"r": 100, "l": 100},
    border=True,
)
# %%
# Patching head to head

SENDER_HEADS= [(0, 4),(0, 3),(0, 1),(24, 7),(0, 10),(0, 12),(24, 5),(19, 0),(20,12),(0,13),(19,11),(34,14),(4,12),(9,9),(19,2),(28,2),(28,5)]
RECEIVER_HEADS = SENDER_HEADS

head_patch_res = []

for s_layer, s_head in SENDER_HEADS:
    for r_layer, r_head in RECEIVER_HEADS:
        if s_layer < r_layer:
            score = path_patch_sender_to_receiver_logits(
                model,
                clean_tokens,
                flipped_tokens,
                sender_head=(s_layer, s_head),
                receiver_head=(r_layer, r_head),
                metric=greater_than_metric_noising
            )
            print(f"Edge {s_layer}.{s_head} -> {r_layer}.{r_head}: {score:.5f}")
            head_patch_res.append({
                'sender': f"{s_layer}.{s_head}",
                'receiver': f"{r_layer}.{r_head}",
                'score': score
            })

#%%
head_patch_res = sorted(head_patch_res, key=lambda x: x['score'], reverse=True)

for item in head_patch_res:
    print(f"Edge {item['sender']} -> {item['receiver']}: {item['score']:.5f}")

# %%
