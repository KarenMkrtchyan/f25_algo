# src/steering/eval_qwen3_clean_corrupt_batch.py
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports (run from src/steering/)
from dataset_gt import build_dataset, split_categories
from activation_steering_qwen import compute_pc1_from_prompts, SteeringContext

torch.set_grad_enabled(False)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def yes_no_ids(tokenizer) -> Tuple[int, int]:
    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]
    return yes_id, no_id


@torch.no_grad()
def margins_for_prompts(model, tokenizer, prompts: List[str], batch_size: int = 8) -> List[float]:
    """margin = logp(Yes) - logp(No) for each prompt (batched)."""
    device = next(model.parameters()).device
    yes_id, no_id = yes_no_ids(tokenizer)

    margins: List[float] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        out = model(**enc)
        logits = out.logits[:, -1, :]  # (B,V)

        logp = F.log_softmax(logits, dim=-1)
        m = (logp[:, yes_id] - logp[:, no_id]).detach().cpu().tolist()
        margins.extend([float(x) for x in m])
    return margins


def parse_heads(s: str) -> List[Tuple[int, int]]:
    """
    Parse heads string like: "17:1,17:11,15:9"
    Returns [(17,1), (17,11), (15,9)]
    """
    out: List[Tuple[int, int]] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad head spec '{p}'. Use L:H like 17:1")
        L, H = p.split(":")
        out.append((int(L), int(H)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=str, default="17:1,17:11,15:9",
                    help="Comma list of layer:head pairs, e.g. '17:1,17:11,15:9'")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low", type=int, default=1000)
    ap.add_argument("--high", type=int, default=9999)
    ap.add_argument("--calib_max", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", type=str, default="../../outputs/qwen3_clean_corrupt_points.csv")
    args = ap.parse_args()

    device = pick_device()
    print("Using device:", device)

    # ------------------------
    # Qwen3-1.7B model
    # ------------------------
    model_id = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    # ------------------------
    # Dataset (clean vs corrupt)
    # ------------------------
    data = build_dataset(n=args.n, seed=args.seed, low=args.low, high=args.high)
    cats = split_categories(data)

    # Keep paired clean/corrupt from same (a,b)
    clean_prompts = [ex.clean for ex in cats["clean_true"]]
    corrupt_prompts = [ex.corrupt for ex in cats["clean_true"]]
    if len(clean_prompts) == 0:
        raise RuntimeError("No clean_true prompts found; try bigger n or different seed.")
    assert len(clean_prompts) == len(corrupt_prompts)

    # Calibration prompts: mix clean+corrupt, then cap
    calib_prompts = []
    for ex in data:
        calib_prompts.append(ex.clean)
        calib_prompts.append(ex.corrupt)
    calib_prompts = calib_prompts[: args.calib_max]

    # Precompute base margins once
    base_clean = margins_for_prompts(model, tokenizer, clean_prompts, batch_size=args.batch_size)
    base_corrupt = margins_for_prompts(model, tokenizer, corrupt_prompts, batch_size=args.batch_size)

    heads = parse_heads(args.heads)

    rows: List[Dict] = []

    for layer, head in heads:
        # Compute PC1 steering direction for THIS (layer, head)
        steering = compute_pc1_from_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=calib_prompts,
            layer_idx=layer,
            heads=(head,),
            max_prompts=len(calib_prompts),
        )
        print(f"Computed steering vectors for L{layer} H{head}")

        def run_condition(prompt_type: str, prompts: List[str], base_margins: List[float], alpha_val: float, direction: str):
            with SteeringContext(model, steering, alpha_head={head: float(alpha_val)}):
                steered = margins_for_prompts(model, tokenizer, prompts, batch_size=args.batch_size)

            for p, bm, sm in zip(prompts, base_margins, steered):
                flipped = int((bm > 0) != (sm > 0))
                rows.append({
                    "model": model_id,
                    "prompt_type": prompt_type,   # clean/corrupt
                    "direction": direction,       # plus/minus
                    "alpha": float(alpha_val),
                    "layer": int(layer),
                    "head": int(head),
                    "prompt": p,
                    "base_margin": float(bm),
                    "steered_margin": float(sm),
                    "margin_shift": float(sm - bm),
                    "flipped": int(flipped),
                    "seed": int(args.seed),
                    "n": int(args.n),
                })

        run_condition("clean", clean_prompts, base_clean, +args.alpha, "plus")
        run_condition("clean", clean_prompts, base_clean, -args.alpha, "minus")
        run_condition("corrupt", corrupt_prompts, base_corrupt, +args.alpha, "plus")
        run_condition("corrupt", corrupt_prompts, base_corrupt, -args.alpha, "minus")

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nSaved points CSV:", args.out_csv)
    print("Tip: plot per (layer,head,prompt_type,direction) from this CSV.")


if __name__ == "__main__":
    main()
