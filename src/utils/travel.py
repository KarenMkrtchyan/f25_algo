import transformer_lens
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

model_name = "Qwen/Qwen3-4B"

try:
    # Try to fetch the model config
    cfg = get_pretrained_model_config(model_name)
    print(f"{model_name} is available in your current TransformerLens installation!")
except ValueError as e:
    print(f"{model_name} is NOT available in your current TransformerLens installation.")
    print("Consider upgrading with: pip install --upgrade transformer-lens")
