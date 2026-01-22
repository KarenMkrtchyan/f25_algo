# src/steering/eval_qwen_fliprate.py
import argparse
import csv
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.activation_steering_qwen import (
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


def make_prompt(a: int, b: int) -> str:
    return f"Is {a} > {b}? Answer:"


@torch.no_grad()
def score_yes_no_logits(
    model,
    tokenizer,
    prompt: str,
) -> Tuple[str, Dict[str, float], float]:
    """
    Returns:
      pred: "Yes" or "No"
      scores: {"Yes": logp, "No": logp}
      margin: logp(Yes) - logp(No)  (positive => Yes)
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :]  # (vocab,)

    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

    logp = F.log_softmax(logits, dim=-1)
    yes_lp = float(logp[yes_id].item())
    no_lp = float(logp[no_id].item())
    margin = yes_lp - no_lp
    pred = "Yes" if margin > 0 else "No"
    return pred, {"Yes": yes_lp, "No": no_lp}, margin


@torch.no_grad()
def score_yes_no_logits_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering,
    alpha_resid: float,
    alpha_head: Dict[int, float],
) -> Tuple[str, Dict[str, float], float]:
    with SteeringContext(model, steering, alpha_resid=alpha_resid, alpha_head=alpha_head):
        return score_yes_no_logits(model, tokenizer, prompt)


def generate_pairs(n: int, low: int, high: int, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        pairs.append((a, b))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of prompts to evaluate")
    ap.add_argument("--low", type=int, default=0, help="min integer")
    ap.add_argument("--high", type=int, default=99, help="max integer")
    ap.add_argument("--seed", type=int, default=42, help="random seed")

    ap.add_argument("--layer", type=int, default=24, help="0-based layer idx for steering")
    ap.add_argument("--head5", type=float, default=4.0, help="alpha for head 5")
    ap.add_argument("--head7", type=float, default=-12.0, help="alpha for head 7")
    ap.add_argument("--alpha_resid", type=float, default=0.0, help="alpha for residual stream")

    ap.add_argument("--calib_n", type=int, default=256, help="number of calibration prompts for PC1")
    ap.add_argument("--out_csv", type=str, default="output/qwen_fliprate.csv", help="output CSV path")
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

    # ---------------------------
    # Calibration prompts for PC1
    # ---------------------------
    # Use a dense grid (stable) rather than random for calibration
    calib_prompts = [make_prompt(a, b) for a in range(50) for b in range(50)]
    print("Calibration prompts total:", len(calib_prompts))

    steering = compute_pc1_from_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=calib_prompts,
        layer_idx=args.layer,
        heads=(5, 7),
        max_prompts=args.calib_n,
    )
    print("Computed steering vectors.")

    # ---------------------------
    # Evaluation dataset
    # ---------------------------
    pairs = generate_pairs(args.n, args.low, args.high, args.seed)

    alpha_head = {5: float(args.head5), 7: float(args.head7)}
    alpha_resid = float(args.alpha_resid)

    # Metrics
    n_total = 0
    n_flip = 0
    n_base_correct = 0
    n_steered_correct = 0
    margin_shifts = []

    # Ensure output directory exists (simple, no pathlib)
    import os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "a", "b", "label",
            "base_pred", "steered_pred",
            "base_yes_logp", "base_no_logp", "base_margin",
            "steered_yes_logp", "steered_no_logp", "steered_margin",
            "margin_shift",
            "flipped",
            "base_correct",
            "steered_correct",
            "alpha_resid", "alpha_head5", "alpha_head7",
            "layer",
        ])

        for i, (a, b) in enumerate(pairs):
            prompt = make_prompt(a, b)
            label = "Yes" if a > b else "No"

            base_pred, base_scores, base_margin = score_yes_no_logits(model, tokenizer, prompt)
            steered_pred, steered_scores, steered_margin = score_yes_no_logits_with_steering(
                model, tokenizer, prompt,
                steering=steering,
                alpha_resid=alpha_resid,
                alpha_head=alpha_head,
            )

            flipped = int(base_pred != steered_pred)
            base_correct = int(base_pred == label)
            steered_correct = int(steered_pred == label)
            margin_shift = steered_margin - base_margin

            n_total += 1
            n_flip += flipped
            n_base_correct += base_correct
            n_steered_correct += steered_correct
            margin_shifts.append(margin_shift)

            w.writerow([
                a, b, label,
                base_pred, steered_pred,
                base_scores["Yes"], base_scores["No"], base_margin,
                steered_scores["Yes"], steered_scores["No"], steered_margin,
                margin_shift,
                flipped,
                base_correct,
                steered_correct,
                alpha_resid, alpha_head[5], alpha_head[7],
                args.layer,
            ])

            if i % 50 == 0:
                print(f"Processed {i}/{args.n}")

    flip_rate = n_flip / max(1, n_total)
    base_acc = n_base_correct / max(1, n_total)
    steered_acc = n_steered_correct / max(1, n_total)

    # Simple summary
    avg_shift = sum(margin_shifts) / max(1, len(margin_shifts))
    print("\n===== SUMMARY =====")
    print("Total:", n_total)
    print("Flip rate:", flip_rate)
    print("Base accuracy:", base_acc)
    print("Steered accuracy:", steered_acc)
    print("Avg margin shift:", avg_shift)
    print("Saved CSV:", args.out_csv)


if __name__ == "__main__":
    main()
