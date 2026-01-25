# src/steering/eval_qwen_clean_corrupt_steering.py
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_gt import build_dataset, split_categories
from activation_steering_qwen import (
    compute_pc1_from_prompts,
    SteeringContext,
)

torch.set_grad_enabled(False)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_margin(model, tokenizer, prompt: str) -> float:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :]

    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

    logp = F.log_softmax(logits, dim=-1)
    return float((logp[yes_id] - logp[no_id]).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="outputs/clean_corrupt_steering.csv")
    args = ap.parse_args()

    device = pick_device()
    print("Using device:", device)

    # ------------------------
    # Load model
    # ------------------------
    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).eval().to(device)

    # ------------------------
    # Build dataset (Arya-style)
    # ------------------------
    data = build_dataset(
        n=args.n,
        seed=args.seed,
        low=1000,
        high=9999,
    )
    cats = split_categories(data)

    clean_prompts = [ex.clean for ex in cats["clean_true"]]
    corrupt_prompts = [ex.corrupt for ex in cats["clean_true"]]

    assert len(clean_prompts) == len(corrupt_prompts)

    # ------------------------
    # Calibration prompts
    # ------------------------
    calib_prompts = []
    for ex in data:
        calib_prompts.append(ex.clean)
        calib_prompts.append(ex.corrupt)

    steering = compute_pc1_from_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=calib_prompts,
        layer_idx=args.layer,
        heads=(args.head,),
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # ------------------------
    # Run experiment
    # ------------------------
    rows = []

    for label, prompts in [
        ("clean", clean_prompts),
        ("corrupt", corrupt_prompts),
    ]:
        for direction in ["plus", "minus"]:
            alpha = args.alpha if direction == "plus" else -args.alpha

            for p in prompts:
                base_margin = score_margin(model, tokenizer, p)

                with SteeringContext(
                    model,
                    steering,
                    alpha_head={args.head: alpha},
                ):
                    steered_margin = score_margin(model, tokenizer, p)

                flipped = int((base_margin > 0) != (steered_margin > 0))

                rows.append({
                    "prompt_type": label,
                    "direction": direction,
                    "alpha": alpha,
                    "base_margin": base_margin,
                    "steered_margin": steered_margin,
                    "margin_shift": steered_margin - base_margin,
                    "flipped": flipped,
                    "layer": args.layer,
                    "head": args.head,
                })

    # ------------------------
    # Save CSV
    # ------------------------
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("Saved:", args.out_csv)

    # Quick summary
    import pandas as pd
    df = pd.DataFrame(rows)

    print("\n=== SUMMARY ===")
    print(df.groupby(["prompt_type", "direction"])["flipped"].mean())
    print(df.groupby(["prompt_type", "direction"])["margin_shift"].mean())


if __name__ == "__main__":
    main()
