import os
import pandas as pd
import sys
import torch as t
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_config import load_model
from utils.device_utils import get_device
from Interpretability import build_dataset, average_logit_tracking, plot_token_logits, get_shared_ylim, ablate_attn_head_last_pos, make_ablated_model
from neel_plotly import imshow
import transformer_lens.utils as utils

dataset = build_dataset(n=100, low=1000, high=9999)

#model_name = "pythia-70m"
model_name = "qwen2.5-3b"
model = load_model(model_name)
device = get_device()

candidate_tokens = [" Yes", " No"]

layer = 24
head  = 5

class AblatedModel:
    def __init__(self, model, layer, head, pos=None):
        self.model = model
        self.layer = layer
        self.head = head
        self.pos = pos

        self.hook = (
            f"blocks.{layer}.attn.hook_result",
            self._ablation_hook,
        )

    def _ablation_hook(self, attn_result, hook):
        if self.pos is None:
            attn_result[:, :, self.head, :] = 0.0
        else:
            attn_result[:, self.pos, self.head, :] = 0.0
        return attn_result

    def __call__(self, *args, **kwargs):
        with self.model.hooks(fwd_hooks=[self.hook]):
            return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

ablated_model = AblatedModel(model, layer, head)

df_clean_avg, df_corrupt_avg = average_logit_tracking(
    model=ablated_model,
    dataset=dataset,
    target_tokens=candidate_tokens,
    token_position=-1,
    max_examples=100,
)

results_folder = "Results"
display_folder = os.path.join(results_folder, "Digit_Experiment")
digit_folder = os.path.join(display_folder, "Logit_Tracking")
output_folder = os.path.join(digit_folder, f"{model_name}")
os.makedirs(output_folder, exist_ok=True)
clean_path_csv = os.path.join(output_folder, f"clean_prompts_ablated_L{layer}H{head}_all_pos.csv")
clean_path_png = os.path.join(output_folder, f"clean_prompts_ablated_L{layer}H{head}_all_pos.png")
corrupt_path_csv = os.path.join(output_folder, f"corrupt_prompts_ablated_L{layer}H{head}_all_pos.csv")
corrupt_path_png = os.path.join(output_folder, f"corrupt_prompts_ablated_L{layer}H{head}_all_pos.png")

ylim = get_shared_ylim(df_clean_avg, df_corrupt_avg)

df_clean_avg.to_csv(clean_path_csv)
df_corrupt_avg.to_csv(corrupt_path_csv)

plot_token_logits(
    df_clean_avg,
    path=clean_path_png
)

plot_token_logits(
    df_corrupt_avg,
    path=corrupt_path_png
)
