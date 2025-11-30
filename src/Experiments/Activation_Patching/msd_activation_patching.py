#%%
import torch
import plotly.express as px
from transformer_lens import HookedTransformer
from transformer_lens.patching import get_act_patch_attn_head_out_by_pos
from data.msd_prompts import generate_msd_dataset

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("qwen2.5-3b", device="cuda")

raw_data = generate_msd_dataset()
tokenized_data = []
target_len = None

for item in raw_data:
    tokens = model.to_tokens(item["clean_prompt"])
    if target_len is None: 
        target_len = tokens.shape[1]
    
    if tokens.shape[1] == target_len:
        tokenized_data.append(item)

print(f"Filtered to {len(tokenized_data)} prompts of length {target_len}")

clean_tokens = torch.cat([model.to_tokens(d["clean_prompt"]) for d in tokenized_data])
corrupted_tokens = torch.cat([model.to_tokens(d["corrupted_prompt"]) for d in tokenized_data])

clean_ans_indices = torch.tensor([model.to_single_token(d["clean_label"]) for d in tokenized_data], device="cuda")
corr_ans_indices = torch.tensor([model.to_single_token(d["corrupted_label"]) for d in tokenized_data], device="cuda")

_, clean_cache = model.run_with_cache(clean_tokens)

def batched_logit_diff_metric(logits):
    last_token_logits = logits[:, -1, :]
    
    # Use 'gather' to pick the specific correct/incorrect token for each row
    clean_logits = last_token_logits.gather(1, clean_ans_indices.unsqueeze(1)).squeeze()
    corr_logits = last_token_logits.gather(1, corr_ans_indices.unsqueeze(1)).squeeze()
    
    return (clean_logits - corr_logits).mean()

print("Running batched patching...")
patch_results = get_act_patch_attn_head_out_by_pos(
    model=model,
    corrupted_tokens=corrupted_tokens,
    clean_cache=clean_cache,
    patching_metric=batched_logit_diff_metric
)

head_scores, _ = patch_results.max(dim=1) 

#%%
fig = px.imshow(
    head_scores.cpu().numpy(),
    labels={"x": "Head", "y": "Layer", "color": "Logit Diff Recovery"},
    title="MSD Circuit: Head Importance",
    color_continuous_scale="RdBu_r",
    origin="lower"
)
fig.show()
# %%
