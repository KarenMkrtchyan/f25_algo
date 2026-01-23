# src/steering/activation_steering_qwen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _pc1(X: torch.Tensor) -> torch.Tensor:
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    v = Vh[0]
    return v / (v.norm() + 1e-12)


@dataclass
class SteeringVectors:
    layer_idx: int
    resid_pc1: torch.Tensor
    head_pc1: Dict[int, torch.Tensor]


@torch.no_grad()
def compute_pc1_from_prompts(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int = 24,
    heads: Tuple[int, int] = (5, 7),
    max_prompts: int = 256,
) -> SteeringVectors:
    device = next(model.parameters()).device
    prompts = prompts[:max_prompts]

    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    n_heads = int(model.config.num_attention_heads)
    hidden = int(model.config.hidden_size)
    head_dim = hidden // n_heads
    if hidden % n_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    buf: Dict[str, torch.Tensor] = {}

    def layer_out_hook(_module, _inputs, output):
        buf["layer_out"] = output.detach()

    def oproj_pre_hook(_module, inputs):
        buf["o_in"] = inputs[0].detach()
        return inputs

    h_layer = layer.register_forward_hook(layer_out_hook)
    h_oproj = attn.o_proj.register_forward_pre_hook(oproj_pre_hook)

    resid_rows: List[torch.Tensor] = []
    head_rows: Dict[int, List[torch.Tensor]] = {h: [] for h in heads}

    try:
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**enc)

            resid = buf["layer_out"][0, -1, :]
            resid_rows.append(resid)

            o_in = buf["o_in"][0, -1, :]
            o_heads = o_in.view(n_heads, head_dim)
            for h in heads:
                head_rows[h].append(o_heads[h].clone())
    finally:
        h_layer.remove()
        h_oproj.remove()

    resid_pc1 = _pc1(torch.stack(resid_rows, dim=0))
    head_pc1 = {h: _pc1(torch.stack(head_rows[h], dim=0)) for h in heads}
    return SteeringVectors(layer_idx=layer_idx, resid_pc1=resid_pc1, head_pc1=head_pc1)


class SteeringContext:
    def __init__(
        self,
        model,
        steering: SteeringVectors,
        alpha_resid: float = 0.0,
        alpha_head: Optional[Dict[int, float]] = None,
    ):
        self.model = model
        self.steering = steering
        self.alpha_resid = float(alpha_resid)
        self.alpha_head = alpha_head or {}
        self._handles = []

    def __enter__(self):
        model = self.model
        device = next(model.parameters()).device

        layer = model.model.layers[self.steering.layer_idx]
        attn = layer.self_attn

        n_heads = int(model.config.num_attention_heads)
        hidden = int(model.config.hidden_size)
        head_dim = hidden // n_heads

        resid_dir = self.steering.resid_pc1.to(device)
        head_dir = {h: v.to(device) for h, v in self.steering.head_pc1.items()}

        def layer_out_hook(_module, _inputs, output):
            if self.alpha_resid == 0.0:
                return output
            out = output.clone()
            out[:, -1, :] += self.alpha_resid * resid_dir.to(out.dtype)
            return out

        def oproj_pre_hook(_module, inputs):
            if not self.alpha_head:
                return inputs

            x = inputs[0]  # (B,S,hidden)
            B, S, H = x.shape
            if H != hidden:
                raise RuntimeError("Unexpected o_proj input size")

            out = x.contiguous().clone()
            last = out[:, -1, :].contiguous().view(B, n_heads, head_dim)

            for h, a in self.alpha_head.items():
                if a == 0.0:
                    continue
                last[:, h, :] = last[:, h, :] + float(a) * head_dir[h].to(last.dtype)

            updated_last = last.reshape(B, hidden).contiguous().clone()
            out2 = torch.cat([out[:, :-1, :], updated_last.unsqueeze(1)], dim=1)
            return (out2,) + tuple(inputs[1:])

        self._handles.append(layer.register_forward_hook(layer_out_hook))
        self._handles.append(attn.o_proj.register_forward_pre_hook(oproj_pre_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        return False


@torch.no_grad()
def score_yes_no(model, tokenizer, prompt: str) -> Tuple[str, Dict[str, float]]:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :]

    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

    logp = F.log_softmax(logits, dim=-1)
    scores = {"Yes": float(logp[yes_id].item()), "No": float(logp[no_id].item())}
    pred = "Yes" if scores["Yes"] > scores["No"] else "No"
    return pred, scores


@torch.no_grad()
def score_yes_no_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering: SteeringVectors,
    alpha_resid: float = 0.0,
    alpha_head: Optional[Dict[int, float]] = None,
) -> Tuple[str, Dict[str, float]]:
    with SteeringContext(model, steering, alpha_resid=alpha_resid, alpha_head=alpha_head):
        return score_yes_no(model, tokenizer, prompt)


@torch.no_grad()
def find_flip(model, tokenizer, prompt: str, steering: SteeringVectors):
    base_pred, _ = score_yes_no(model, tokenizer, prompt)

    search_alphas = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
    search_heads = [-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0]

    for ar in search_alphas:
        for a5 in search_heads:
            for a7 in search_heads:
                pred, scores = score_yes_no_with_steering(
                    model, tokenizer, prompt,
                    steering=steering,
                    alpha_resid=ar,
                    alpha_head={5: a5, 7: a7},
                )
                if pred != base_pred:
                    return (ar, a5, a7, pred, scores)
    return None


@torch.no_grad()
def append_forced_choice(
    model,
    tokenizer,
    prompt: str,
    steering: SteeringVectors,
    alpha_resid: float,
    alpha_head: Dict[int, float],
) -> str:
    """
    One-step decode:
      - compute next-token logits under steering
      - choose Yes/No by logprob
      - append that token to the prompt and return the text

    This makes the printed answer match the flipped logits.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    with SteeringContext(model, steering, alpha_resid=alpha_resid, alpha_head=alpha_head):
        out = model(**enc)
        logits = out.logits[:, -1, :]  # (B,V)

    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

    chosen = yes_id if logits[0, yes_id] > logits[0, no_id] else no_id
    new_ids = torch.cat([enc["input_ids"], torch.tensor([[chosen]], device=device)], dim=1)
    return tokenizer.decode(new_ids[0], skip_special_tokens=True)
