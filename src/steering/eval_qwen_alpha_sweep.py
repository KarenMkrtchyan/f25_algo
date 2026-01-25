# src/steering/eval_qwen_alpha_sweep.py
import argparse
import csv
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.activation_steering_qwen import compute_pc1_from_prompts, SteeringContext
from steering.dataset_gt import build_dataset, split_categories
from steering.scoring_yesno import score_yes_no_margin


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
    ap.add_argument("--category", type=str, default="clean_true",
                    choices=["clean_true", "clean_false", "corrupt_true", "corrupt_false"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12])
    ap.add_argument("--out_csv", type=str, default="outputs/alpha_sweep.csv")
    args = ap.parse_args()

    device = pick_device()

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    ).eval().to(device)

    data = build_dataset(n=100)
    cats = split_categories(data)
    prompts = [
        ex.clean if args.category.startswith("clean") else ex.corrupt
        for ex in cats[args.category]
    ]

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

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mean_margin"])

        for a in args.alphas:
            margins = []
            for p in prompts:
                with SteeringContext(model, steering, alpha_head={args.head: a}):
                    _, _, m = score_yes_no_margin(model, tokenizer, p)
                margins.append(m)

            w.writerow([a, sum(margins) / len(margins)])

    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
