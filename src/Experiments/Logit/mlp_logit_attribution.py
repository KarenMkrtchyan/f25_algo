#%%
import pandas as pd
from transformer_lens.HookedTransformer import HookedTransformer

def neuron_logit_attribution(model, layers, yes_tok=" yes", no_tok=" NO"):
    yes_id = model.to_single_token(yes_tok)
    no_id = model.to_single_token(no_tok)
    truth_dir = model.W_U[:, yes_id] - model.W_U[:, no_id]
    
    results = []
    
    for layer in layers:
        W_out = model.blocks[layer].mlp.W_out
        
        # score for every neuron in this layer
        scores = W_out @ truth_dir
        
        for i, score in enumerate(scores.detach().cpu().numpy()):
            results.append({"layer": layer, "neuron": i, "score": score})

    df = pd.DataFrame(results)
    df["abs_score"] = df["score"].abs() 
    return df.sort_values("abs_score", ascending=False).head(20)

model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3b")
#%%
top_neurons = neuron_logit_attribution(model, [19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34])

print(top_neurons)

# %%
