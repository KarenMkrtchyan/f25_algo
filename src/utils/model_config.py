from .Setup import Dict, Any, Optional, HookedTransformer, HookedTransformerConfig, t, device

COMMON_MODELS = {
    # GPT-2 Family
    "gpt2": "gpt2",
    "gpt2-small": "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    
    # GPT-Neo and GPT-J
    "gpt-neo-125M": "EleutherAI/gpt-neo-125M",
    "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6B": "EleutherAI/gpt-j-6b",

    # OPT Family
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",

    # Pythia Family
    "pythia-70m": "EleutherAI/pythia-70m-deduped",
    "pythia-160m": "EleutherAI/pythia-160m-deduped",
    "pythia-410m": "EleutherAI/pythia-410m-deduped",
    "pythia-1b": "EleutherAI/pythia-1b-deduped",
    "pythia-1.4b": "EleutherAI/pythia-1.4b-deduped",
    "pythia-2.8b": "EleutherAI/pythia-2.8b-deduped",
    "pythia-6.9b": "EleutherAI/pythia-6.9b-deduped",
    "pythia-12b": "EleutherAI/pythia-12b-deduped",

    # LLaMA Family (Meta)
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "llama-70b": "meta-llama/Llama-2-70b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-8b-it": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",

    # Gemma Family (Google)
    "gemma-2b": "google/gemma-2b",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-7b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",

    # Qwen Family (Alibaba)
    "qwen-7b": "Qwen/Qwen-7B",
    "qwen-14b": "Qwen/Qwen-14B",
    "qwen-72b": "Qwen/Qwen-72B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2-72b": "Qwen/Qwen2-72B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B", 

    # Phi Family (Microsoft)
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    if model_name not in COMMON_MODELS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(COMMON_MODELS.keys())}")
    
    config = COMMON_MODELS[model_name]
    if isinstance(config, str):
        return {"model_name": config}
    else:
        return config.copy()

def load_model(
    model_name: str, 
    target_device: Optional[str] = None,
    **kwargs
) -> HookedTransformer:
    if target_device is None:
        target_device = device
    
    if model_name in COMMON_MODELS:
        config = get_model_config(model_name)
        
        if "model_name" in config:
            model = HookedTransformer.from_pretrained(
                config["model_name"],
                device=target_device,
                **kwargs
            )
        else:
            cfg = HookedTransformerConfig(**config)
            model = HookedTransformer(cfg, device=target_device)
    else:
        try:
            model = HookedTransformer.from_pretrained(
                model_name,
                device=target_device,
                **kwargs
            )
        except Exception as e:
            raise ValueError(f"Could not load model '{model_name}'. Error: {e}")
    
    return model

def create_custom_model(
    n_layers: int,
    d_model: int,
    n_ctx: int,
    d_head: int,
    n_heads: int = -1,
    d_mlp: int = -1,
    d_vocab: int = 50257,
    act_fn: str = "gelu_new",
    attention_dir: str = "causal",
    attn_only: bool = False,
    use_attn_result: bool = False,
    normalization_type: str = "LNPre",
    positional_embedding_type: str = "standard",
    target_device: Optional[str] = None,
    **kwargs
) -> HookedTransformer:
    
    if target_device is None:
        target_device = device
    
    config = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        act_fn=act_fn,
        attention_dir=attention_dir,
        attn_only=attn_only,
        use_attn_result=use_attn_result,
        normalization_type=normalization_type,
        positional_embedding_type=positional_embedding_type,
        device=target_device,
        **kwargs
    )
    
    return HookedTransformer(config)

def get_induction_score_store(n_layers: int, n_heads: int, device: str = "cpu") -> t.Tensor:
    """
    Create a tensor to store induction scores.
    
    Args:
        n_layers: Number of layers in the model
        n_heads: Number of attention heads per layer
        device: Device to create the tensor on
        
    Returns:
        Zero-initialized tensor for storing induction scores
    """
    return t.zeros((n_layers, n_heads), device=device)

def list_available_models() -> list:
    return list(COMMON_MODELS.keys())

def get_model_info(model_name: str) -> Dict[str, Any]:
    if model_name not in COMMON_MODELS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(COMMON_MODELS.keys())}")
    
    config = COMMON_MODELS[model_name]
    
    if isinstance(config, str):
        return {
            "name": model_name,
            "type": "pretrained",
            "model_name": config,
            "description": f"Pretrained {config} model",
            "is_pretrained": True,
            "config_source": "huggingface"
        }
    else:
        return {
            "name": model_name,
            "type": "custom",
            "config": config,
            "description": f"Custom model configuration: {model_name}",
            "is_pretrained": False,
            "config_source": "custom",
            "architecture": {
                "layers": config.get("n_layers", "unknown"),
                "model_dim": config.get("d_model", "unknown"),
                "heads": config.get("n_heads", "unknown"),
                "head_dim": config.get("d_head", "unknown"),
                "context_length": config.get("n_ctx", "unknown"),
                "vocab_size": config.get("d_vocab", "unknown"),
                "attention_only": config.get("attn_only", False),
                "activation": config.get("act_fn", "unknown"),
                "normalization": config.get("normalization_type", "unknown"),
                "positional_embedding": config.get("positional_embedding_type", "unknown")
            }
        }

def get_model_summary(model_name: str) -> str:
    info = get_model_info(model_name)
    
    if info["is_pretrained"]:
        return f"{info['name']}: {info['description']} (from {info['config_source']})"
    else:
        arch = info["architecture"]
        return f"{info['name']}: Custom model with {arch['layers']} layers, {arch['model_dim']}d model, {arch['heads']} heads"

if __name__ == "__main__":
    print("=== Model Setup File Demo ===\n")
    
    print("Available models:")
    for model_name in list_available_models():
        summary = get_model_summary(model_name)
        print(f"  {summary}")
    
    print(f"\nUsing device: {device}")
    
    # Test get_model_info function
    print("\n=== Testing get_model_info Function ===")
    
    # Test with pretrained model
    print("\nGPT-2 Small info:")
    gpt2_info = get_model_info("gpt2-small")
    print(f"  Type: {gpt2_info['type']}")
    print(f"  Pretrained: {gpt2_info['is_pretrained']}")
    print(f"  Source: {gpt2_info['config_source']}")
    print(f"  Description: {gpt2_info['description']}")
    
    print("\n=== Testing Model Loading ===")
    try:
        model = load_model("pythia-70m")
        print(f"✓ Loaded model: {model.cfg.model_name}")
        print(f"  - Layers: {model.cfg.n_layers}")
        print(f"  - Heads: {model.cfg.n_heads}")
        print(f"  - Model dimension: {model.cfg.d_model}")
        print(f"  - Context length: {model.cfg.n_ctx}")
        print(f"  - Device: {model.cfg.device}")
        
        test_tokens = model.to_tokens("Hello world!")
        print(f"  - Test tokens shape: {test_tokens.shape}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    print("\n=== Model Setup Complete ===")
