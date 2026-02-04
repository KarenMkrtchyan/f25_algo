#%%
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import plotly.express as px
from data.act_prompts import prompt_list
from Interpretability.path_patching import act_patch, Node, IterNode, imshow, hist
from transformer_lens.HookedTransformer import HookedTransformer
import torch as t
from torch import Tensor
from jaxtyping import Float, Int, Bool
from dotenv import load_dotenv
from huggingface_hub import login

# %%
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
t.set_grad_enabled(False)

load_dotenv()
hf_key = os.getenv("HF_TOKEN")
login(token=hf_key)

#%%

model_name = "EleutherAI/pythia-70m-deduped"

model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    torch_dtype=t.bfloat16,
)

model.set_use_split_qkv_input(True)
#%%

prompt_list=prompt_list[:20]

#prompts = [p["clean_prompt"] for p in prompt_list]
prompts = [p["clean_prompt"] + " " for p in prompt_list]    # Phi

labels = [p["clean_label"] for p in prompt_list]

# Define the answers for each prompt, in the form (correct, incorrect)
#%%
answers = [(" Yes", " No") if label == " Yes" else (" No", " Yes") for label in labels]

# Define the answer tokens (same shape as the answers)
yes_id = model.to_single_token(" Yes")
no_id  = model.to_single_token(" No")

answer_tokens = []
for label in labels:
    if label == " Yes":
        answer_tokens.append([yes_id, no_id])
    else:
        answer_tokens.append([no_id, yes_id])
answer_tokens = t.tensor(answer_tokens, device=device)

#%%

def patching_filter(name):
    if any(n in name for n in ["resid_pre", "attn_out", "mlp_out", "z"]):
        return True

    if "hook_q" in name and ".0." in name:
        return True
    
    return False

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

with t.inference_mode():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens,names_filter=patching_filter)
    flipped_logits, flipped_cache = model.run_with_cache(flipped_tokens,names_filter=patching_filter)

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

labels =[f"{tok} ({i})" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]

# %%

results = act_patch(
    model=model,
    orig_input=flipped_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=greater_than_metric_noising,
    verbose=True,
)

#%%
assert results.keys() == {"resid_pre", "attn_out", "mlp_out"}

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "4digit")
output_folder = os.path.join(digit_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
block_path = os.path.join(output_folder, "AP_blocks.png")
head_path = os.path.join(output_folder, "AP_heads.png")

fig_blocks = imshow(
    t.stack([r.T for r in results.values()]) * 100,
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    labels={"x": "Sequence position", "y": "Layer", "color": "Logit diff variation"},
    x=labels,
    xaxis_tickangle=45,
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1300,
    margin={"r": 100, "l": 100},
    return_fig = True
)
fig_blocks.write_image(block_path)
# %% Attnetion head output

results = act_patch(
    model=model,
    orig_input=flipped_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["z"]),
    patching_metric=greater_than_metric_noising,
    verbose=True,
)
#%% 

fig_heads = imshow(
    results['z'] * 100,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=600,
    margin={"r": 100, "l": 100},
    return_fig = True
)
fig_heads.write_image(head_path)

# %%
hist_res = results['z'] * 100
hist(hist_res.detach().cpu().flatten())

# %%
