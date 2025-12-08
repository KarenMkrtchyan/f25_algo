import os
import torch as t


def get_device():
    """
    Decide which device to use.

    - On Lambda (FORCE_CUDA=1 set in env): use CUDA if available, else error/fallback.
    - Locally (no FORCE_CUDA): prefer MPS, then CPU. We *don’t* try to use CUDA here.
    """
    # Case 1: Running on Lambda GPU (we'll set this env var from run_on_lambda.py)
    if os.environ.get("FORCE_CUDA") == "1":
        if t.cuda.is_available():
            return "cuda"
        else:
            # If this ever prints, something is wrong with the Lambda setup
            print("⚠️ FORCE_CUDA=1 but torch.cuda.is_available() is False. Falling back to CPU.")
            return "cpu"

    # Case 2: Local dev (Mac etc.) – no FORCE_CUDA
    if hasattr(t.backends, "mps") and t.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"