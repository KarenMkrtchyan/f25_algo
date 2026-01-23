# src/steering/eval_qwen_steered.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.activation_steering_qwen import (
    compute_pc1_from_prompts,
    score_yes_no,
    score_yes_no_with_steering,
    find_flip,
    append_forced_choice,   # <-- IMPORTANT
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


def main():
    # --------------------------------------------------
    # 1. Setup
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2. Calibration prompts (for PCA)
    # --------------------------------------------------
    calib_prompts = [
        make_prompt(a, b)
        for a in range(50)
        for b in range(50)
    ]
    print(f"Calibration prompts: {len(calib_prompts)}")

    # --------------------------------------------------
    # 3. Compute steering vectors
    # --------------------------------------------------
    steering = compute_pc1_from_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=calib_prompts,
        layer_idx=24,
        heads=(5, 7),
        max_prompts=256,
    )
    print("Computed steering vectors.")

    # --------------------------------------------------
    # 4. Test example
    # --------------------------------------------------
    prompt = make_prompt(2, 7)
    print("\nPROMPT:", prompt)

    # Base logits
    base_pred, base_scores = score_yes_no(model, tokenizer, prompt)
    print("BASE (logits):", base_pred, base_scores)

    # --------------------------------------------------
    # 5. Search for a flip
    # --------------------------------------------------
    print("\nRunning flip search...")
    res = find_flip(model, tokenizer, prompt, steering)

    if res is None:
        print("No flip found in search grid.")
        return

    alpha_resid, a5, a7, flipped_pred, flipped_scores = res

    print(
        f"\nFOUND FLIP:\n"
        f"  alpha_resid = {alpha_resid}\n"
        f"  head 5      = {a5}\n"
        f"  head 7      = {a7}\n"
        f"  prediction  = {flipped_pred}\n"
        f"  scores      = {flipped_scores}"
    )

    # --------------------------------------------------
    # 6. (Optional) Print text that matches the flipped logits
    # --------------------------------------------------
    forced_text = append_forced_choice(
        model,
        tokenizer,
        prompt,
        steering,
        alpha_resid=alpha_resid,
        alpha_head={5: a5, 7: a7},
    )

    print("\nTEXT OUTPUT (forced to match logits):")
    print(forced_text)


if __name__ == "__main__":
    main()
