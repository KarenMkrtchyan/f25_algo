# src/steering/eval_hf_attn_output_steering_clean_corrupt.py
import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)


@dataclass
class Example:
    a: int
    b: int
    clean: str
    corrupt: str
    label_clean: str
    label_corrupt: str


def make_prompt(a: int, b: int) -> str:
    return f"Is {a} > {b}? Answer:"


def build_dataset(n: int, seed: int, low: int, high: int) -> List[Example]:
    rng = random.Random(seed)
    out: List[Example] = []
    for _ in range(n):
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        clean = make_prompt(a, b)
        corrupt = make_prompt(b, a)
        out.append(
            Example(
                a=a,
                b=b,
                clean=clean,
                corrupt=corrupt,
                label_clean=("Yes" if a > b else "No"),
                label_corrupt=("Yes" if b > a else "No"),
            )
        )
    return out


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    for y in [" Yes", "Yes"]:
        ids = tokenizer(y, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            yes_id = ids[0]
            break
    else:
        raise ValueError("Could not find single-token 'Yes'")

    for n in [" No", "No"]:
        ids = tokenizer(n, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            no_id = ids[0]
            break
    else:
        raise ValueError("Could not find single-token 'No'")

    return yes_id, no_id


@torch.no_grad()
def margin_yes_no(model, tokenizer, prompt: str, yes_id: int, no_id: int) -> float:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :]
    logp = F.log_softmax(logits, dim=-1)
    return float((logp[yes_id] - logp[no_id]).item())


def get_layer(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    raise ValueError("Unsupported model structure: expected model.model.layers")


@torch.no_grad()
def compute_attn_out_direction(
    model,
    tokenizer,
    examples: List[Example],
    layer_idx: int,
    prompt_type: str,
    max_calib: int,
) -> torch.Tensor:
    """
    Direction = mean(attn_output | label=Yes) - mean(attn_output | label=No)
    at last token. attn_output is the output of layer.self_attn forward (shape hidden).
    """
    device = next(model.parameters()).device
    layer = get_layer(model, layer_idx)
    attn = layer.self_attn

    buf: Dict[str, torch.Tensor] = {}

    def attn_out_hook(_module, _inputs, output):
        # output can be Tensor or tuple depending on model; handle both
        out = output[0] if isinstance(output, (tuple, list)) else output
        buf["attn_out"] = out.detach()
        return output

    h = attn.register_forward_hook(attn_out_hook)

    yes_vecs = []
    no_vecs = []
    try:
        for ex in examples[:max_calib]:
            if prompt_type == "clean":
                prompt = ex.clean
                label = ex.label_clean
            else:
                prompt = ex.corrupt
                label = ex.label_corrupt

            enc = tokenizer(prompt, return_tensors="pt").to(device)
            _ = model(**enc)

            v = buf["attn_out"][0, -1, :].to("cpu")  # (hidden,)
            if label == "Yes":
                yes_vecs.append(v)
            else:
                no_vecs.append(v)

        if len(yes_vecs) == 0 or len(no_vecs) == 0:
            raise RuntimeError("Calibration produced empty Yes/No group. Increase calib_n or widen range.")

        yes_mean = torch.stack(yes_vecs).mean(dim=0)
        no_mean = torch.stack(no_vecs).mean(dim=0)
        direction = yes_mean - no_mean
        direction = direction / (direction.norm() + 1e-8)
        return direction
    finally:
        h.remove()


class AttnOutSteeringContext:
    """
    Adds alpha * direction to the attention output at the last token of a given layer.
    """
    def __init__(self, model, layer_idx: int, direction: torch.Tensor, alpha: float):
        self.model = model
        self.layer_idx = layer_idx
        self.direction = direction
        self.alpha = float(alpha)
        self._handle = None

    def __enter__(self):
        device = next(self.model.parameters()).device
        layer = get_layer(self.model, self.layer_idx)
        attn = layer.self_attn
        dir_vec = self.direction.to(device)

        def attn_out_hook(_module, _inputs, output):
            # output: (B,S,H) or tuple
            if isinstance(output, (tuple, list)):
                out0 = output[0].clone()
                out0[:, -1, :] = out0[:, -1, :] + self.alpha * dir_vec.to(out0.dtype)
                return (out0,) + tuple(output[1:])
            else:
                out = output.clone()
                out[:, -1, :] = out[:, -1, :] + self.alpha * dir_vec.to(out.dtype)
                return out

        self._handle = attn.register_forward_hook(attn_out_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
        self._handle = None
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, type=str)

    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--head", default=-1, type=int, help="for labeling only (keeps your naming)")
    ap.add_argument("--alpha", default=10.0, type=float)

    ap.add_argument("--n", default=100, type=int)
    ap.add_argument("--calib_n", default=200, type=int)

    ap.add_argument("--low", default=0, type=int)
    ap.add_argument("--high", default=99, type=int)
    ap.add_argument("--seed", default=42, type=int)

    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"])
    args = ap.parse_args()

    device = pick_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    yes_id, no_id = yes_no_token_ids(tokenizer)

    data = build_dataset(args.n, args.seed, args.low, args.high)
    calib = build_dataset(max(args.calib_n, args.n), args.seed + 999, args.low, args.high)

    dir_clean = compute_attn_out_direction(model, tokenizer, calib, args.layer, "clean", args.calib_n)
    dir_corrupt = compute_attn_out_direction(model, tokenizer, calib, args.layer, "corrupt", args.calib_n)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    alpha_mag = abs(float(args.alpha))

    for prompt_type in ["clean", "corrupt"]:
        direction_vec = dir_clean if prompt_type == "clean" else dir_corrupt

        for direction_name, alpha_signed in [("minus", -alpha_mag), ("plus", alpha_mag)]:
            for ex in data:
                prompt = ex.clean if prompt_type == "clean" else ex.corrupt
                label = ex.label_clean if prompt_type == "clean" else ex.label_corrupt

                base_margin = margin_yes_no(model, tokenizer, prompt, yes_id, no_id)

                with AttnOutSteeringContext(model, args.layer, direction_vec, alpha_signed):
                    steered_margin = margin_yes_no(model, tokenizer, prompt, yes_id, no_id)

                rows.append(
                    dict(
                        model_id=args.model_id,
                        layer=int(args.layer),
                        head=int(args.head),            # stored for naming consistency
                        alpha=float(alpha_mag),
                        direction=direction_name,
                        prompt_type=prompt_type,
                        a=int(ex.a),
                        b=int(ex.b),
                        label=label,
                        base_margin=float(base_margin),
                        steered_margin=float(steered_margin),
                        margin_shift=float(steered_margin - base_margin),
                        flipped=int((base_margin > 0) != (steered_margin > 0)),
                    )
                )
                if len(rows) % 50 == 0:
                    print(f"Progress: {len(rows)} rows written...")


    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Saved points CSV:", args.out_csv)


if __name__ == "__main__":
    main()
