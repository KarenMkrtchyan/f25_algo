# src/steering/eval_qwen_clean_corrupt_steering_batch.py
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# LOCAL imports (works when you run from src/steering/)
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
    """
    Returns margin = logp(Yes) - logp(No) for each prompt (batched).
    """
    device = next(model.parameters()).device
    yes_id, no_id = yes_no_ids(tokenizer)

    margins: List[float] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        out = model(**enc)
        logits = out.logits[:, -1, :]  # (B, V)

        logp = F.log_softmax(logits, dim=-1)
        m = (logp[:, yes_id] - logp[:, no_id]).detach().cpu().tolist()
        margins.extend([float(x) for x in m])

    return margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low", type=int, default=1000)
    ap.add_argument("--high", type=int, default=9999)

    ap.add_argument("--calib_max", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", type=str, default="../../outputs/clean_corrupt_steering_points.csv")
    args = ap.parse_args()

    device = pick_device()
    print("Using device:", device)

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    # ------------------------
    # Dataset (Arya builder)
    # ------------------------
    data = build_dataset(n=args.n, seed=args.seed, low=args.low, high=args.high)
    cats = split_categories(data)

    # IMPORTANT: use SAME pairs for clean and corrupt. Use clean_true bucket as you were doing.
    clean_prompts = [ex.clean for ex in cats["clean_true"]]
    corrupt_prompts = [ex.corrupt for ex in cats["clean_true"]]

    if len(clean_prompts) == 0:
        raise RuntimeError("No clean_true prompts found. Try a different seed or larger n.")
    assert len(clean_prompts) == len(corrupt_prompts)

    # ------------------------
    # Calibration prompts
    # ------------------------
    calib_prompts = []
    for ex in data:
        calib_prompts.append(ex.clean)
        calib_prompts.append(ex.corrupt)
    calib_prompts = calib_prompts[: args.calib_max]

    steering = compute_pc1_from_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=calib_prompts,
        layer_idx=args.layer,
        heads=(args.head,),
        max_prompts=len(calib_prompts),
    )
    print("Computed steering vectors.")

    # ------------------------
    # Compute base margins (batched)
    # ------------------------
    base_clean = margins_for_prompts(model, tokenizer, clean_prompts, batch_size=args.batch_size)
    base_corrupt = margins_for_prompts(model, tokenizer, corrupt_prompts, batch_size=args.batch_size)

    # ------------------------
    # Compute steered margins for +alpha and -alpha
    # ------------------------
    rows: List[Dict] = []

    def run_condition(prompt_type: str, prompts: List[str], base_margins: List[float], alpha_val: float, direction: str):
        with SteeringContext(model, steering, alpha_head={args.head: float(alpha_val)}):
            steered = margins_for_prompts(model, tokenizer, prompts, batch_size=args.batch_size)

        for p, bm, sm in zip(prompts, base_margins, steered):
            flipped = int((bm > 0) != (sm > 0))
            rows.append({
                "prompt_type": prompt_type,      # clean / corrupt
                "direction": direction,          # plus / minus
                "alpha": float(alpha_val),
                "prompt": p,
                "base_margin": float(bm),
                "steered_margin": float(sm),
                "margin_shift": float(sm - bm),
                "flipped": int(flipped),
                "layer": int(args.layer),
                "head": int(args.head),
                "seed": int(args.seed),
                "n": int(args.n),
            })

    run_condition("clean", clean_prompts, base_clean, +args.alpha, "plus")
    run_condition("clean", clean_prompts, base_clean, -args.alpha, "minus")
    run_condition("corrupt", corrupt_prompts, base_corrupt, +args.alpha, "plus")
    run_condition("corrupt", corrupt_prompts, base_corrupt, -args.alpha, "minus")

    # ------------------------
    # Save CSV
    # ------------------------
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Quick printed summary
    def summarize(pt: str, d: str):
        xs = [r for r in rows if r["prompt_type"] == pt and r["direction"] == d]
        flip_rate = sum(r["flipped"] for r in xs) / max(1, len(xs))
        mean_shift = sum(r["margin_shift"] for r in xs) / max(1, len(xs))
        print(f"{pt:7s} {d:5s} | flip_rate={flip_rate:.3f} | mean_shift={mean_shift:.3f}")

    print("\n=== SUMMARY ===")
    summarize("clean", "plus")
    summarize("clean", "minus")
    summarize("corrupt", "plus")
    summarize("corrupt", "minus")

    print("\nSaved points CSV:", args.out_csv)


if __name__ == "__main__":
    main()
