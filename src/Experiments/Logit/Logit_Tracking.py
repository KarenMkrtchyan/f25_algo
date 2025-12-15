import sys
import os
import pandas as pd
import torch as t
from torch.nn import functional as F
import transformer_lens.utils as Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from Interpretability import logit_lens_df, track_tokens_df, plot_token_logits, test_logit_lens

models = ["qwen2.5-3b"]
#prompts = ["After John and Mary went to the store, John gave a bottle of milk to"]
prompts = ["Is 7 > 3? Answer:", "Is 56 > 78? Answer:", "Is 512 > 678? Answer:", "Is 6789 > 5678? Answer:", "Is 9123562356280366 > 3610617930546521? Answer:"]

for model_name in models:
    model = load_model(model_name)
    print("\n")

    for text in prompts:
        candidates = [" Yes", " No"]

        tokens = model.to_tokens(text, prepend_bos=False)
        print("Tokens:", tokens)
        print("Decoded:", model.to_str_tokens(tokens))
        logits, _ = model.run_with_cache(tokens)
        next_token_id = logits[0, -1].argmax(dim = -1)
        print("Next token ID:", next_token_id.item())
        print("Next token string:", model.to_str_tokens(next_token_id))
        print("\n")

        
        df_tokens = track_tokens_df(model, text, candidates)
        test_logit_lens(df_tokens, candidates)

        results_folder = "Results"
        display_folder = os.path.join(results_folder, "Logit_Tracking")
        output_folder = os.path.join(display_folder, f"{model_name}")
        os.makedirs(output_folder, exist_ok=True)
        plot_path = os.path.join(output_folder, f"Interesting{len(text)}--{candidates}.png")
        csv_path = os.path.join(output_folder, f"Interesting{len(text)}--{candidates}.csv")

        df_tokens.to_csv(csv_path, index=False)

        plot_token_logits(df_tokens, path = plot_path)