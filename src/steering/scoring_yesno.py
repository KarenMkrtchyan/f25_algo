# src/steering/scoring_yesno.py
from __future__ import annotations
from typing import Tuple, Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_yes_no_margin(
    model,
    tokenizer,
    prompt: str,
) -> Tuple[str, Dict[str, float], float]:
    """
    Returns:
      pred: "Yes" or "No"
      scores: {"Yes": logp, "No": logp}
      margin: logp(Yes) - logp(No)
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :]

    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id  = tokenizer("No",  add_special_tokens=False)["input_ids"][0]

    logp = F.log_softmax(logits, dim=-1)
    yes_lp = float(logp[yes_id].item())
    no_lp  = float(logp[no_id].item())
    margin = yes_lp - no_lp

    pred = "Yes" if margin > 0 else "No"
    return pred, {"Yes": yes_lp, "No": no_lp}, margin
