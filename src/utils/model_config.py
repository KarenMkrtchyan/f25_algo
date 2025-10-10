import utils.Setup
from utils.Setup import Dict, Any, Optional, HookedTransformer, HookedTransformerConfig, t

# Model configuration constants
MODEL_CONFIGS = {
    "gpt2_small_custom": {
        "d_model": 768,
        "d_head": 64,
        "n_heads": 12,
        "n_layers": 2,
        "n_ctx": 2048,
        "d_vocab": 50278,
        "attention_dir": "causal",
        "attn_only": True,
        "tokenizer_name": "EleutherAI/gpt-neox-20b",
        "seed": 398,
        "use_attn_result": True,
        "normalization_type": None,
        "positional_embedding_type": "shortformer",
    },
    "gpt2_small": {
        # Default GPT-2 small configuration
        "model_name": "gpt2-small",
    },
    "gpt2_medium": {
        "model_name": "gpt2-medium",
    },
    "pythia_70m": {
        "model_name": "EleutherAI/pythia-70m-deduped",
    }
}

def get_model_config(config_name: str) -> Dict[str, Any]:
    """
    Get model configuration by name.
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        Dictionary containing model configuration parameters
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available configs: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[config_name].copy()

def load_model(config_name: str, device: Optional[str] = None) -> HookedTransformer:
    """
    Load a model using the specified configuration.
    
    Args:
        config_name: Name of the configuration to use
        device: Device to load the model on (optional)
        
    Returns:
        Loaded HookedTransformer model
    """
    config = get_model_config(config_name)
    
    if "model_name" in config:
        # Load pretrained model
        model = HookedTransformer.from_pretrained(
            config["model_name"],
            device=device
        )
    else:
        # Create model from custom configuration
        cfg = HookedTransformerConfig(**config)
        model = HookedTransformer(cfg, device=device)
    
    return model

# Example usage and testing
if __name__ == "__main__":
    print("Available model configurations:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name}: {config}")
    
    # Example: Load a model
    try:
        model = load_model("gpt2_small")
        print(f"\nLoaded model: {model.cfg.model_name}")
        print(f"Model device: {model.cfg.device}")
    except Exception as e:
        print(f"Error loading model: {e}")

    