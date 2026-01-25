# src/steering/eval_qwen_scatter.py
import argparse
import csv
import os
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steering_qwen import (
    compute_pc1_from_prompts,
    SteeringContext,
)
from dataset_gt import build_dataset, split_categories
from scoring_yesno import score_yes_no_margin


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--category", type=str, default="clean_true",
                    choices=["clean_true", "clean_false", "corrupt_true", "corrupt_false"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, default="outputs/base_vs_steered_margin.png")
    args = ap.parse_args()

    device = pick_device()

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=None,
    ).eval().to(device)

    # dataset
    data = build_dataset(n=args.n)
    cats = split_categories(data)
    prompts = [
        ex.clean if args.category.startswith("clean") else ex.corrupt
        for ex in cats[args.category]
    ]

    # calibration prompts
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

    base_margins = []
    steered_margins = []

    for p in prompts:
        _, _, base = score_yes_no_margin(model, tokenizer, p)
        with SteeringContext(model, steering, alpha_head={args.head: args.alpha}):
            _, _, steered = score_yes_no_margin(model, tokenizer, p)

        base_margins.append(base)
        steered_margins.append(steered)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure()
    plt.scatter(base_margins, steered_margins, s=10, alpha=0.6)
    lims = [
        min(base_margins + steered_margins),
        max(base_margins + steered_margins),
    ]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("base_margin = logp(Yes) - logp(No)")
    plt.ylabel("steered_margin = logp(Yes) - logp(No)")
    plt.title(f"Layer {args.layer} Head {args.head} α={args.alpha} ({args.category})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
