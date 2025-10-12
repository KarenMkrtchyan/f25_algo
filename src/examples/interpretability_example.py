#!/usr/bin/env python3
"""
Example usage of interpretability functions.
This script demonstrates how to use the interpretability functions in practice.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch as t
from utils import load_model
from Interpretability.Functions import (
    hook_function,
    get_induction_score_store,
    head_zero_ablation_hook,
    head_mean_ablation_hook,
    logit_attribution
)
from Interpretability.Attention_display import attention_head_display, attention_pattern_display


def example_basic_functions():
    """Demonstrate basic interpretability functions."""
    print("🔬 Basic Interpretability Functions Example")
    print("=" * 50)
    
    # Create some test data
    batch_size, seq_len, n_heads, d_head = 1, 8, 4, 16
    device = "cpu"
    
    # Create attention pattern
    attn_pattern = t.randn(batch_size, n_heads, seq_len, seq_len)
    print(f"✓ Created attention pattern: {attn_pattern.shape}")
    
    # Test hook function
    mock_hook = type('MockHook', (), {'layer': lambda: 2})()
    result = hook_function(None, attn_pattern, mock_hook)
    print(f"✓ Hook function result: {result.shape} (should be same as input)")
    
    # Test induction score store
    score_store = get_induction_score_store(6, 8, device)
    print(f"✓ Induction score store: {score_store.shape}")
    print(f"  All values are zero: {t.all(score_store == 0)}")
    
    # Test ablation hooks
    z_tensor = t.randn(batch_size, seq_len, n_heads, d_head)
    print(f"✓ Created z tensor: {z_tensor.shape}")
    
    # Zero ablation
    z_copy = z_tensor.clone()
    head_zero_ablation_hook(z_copy, mock_hook, 0)
    print(f"✓ Zero ablation - head 0 is zero: {t.all(z_copy[:, :, 0, :] == 0)}")
    
    # Mean ablation
    z_copy = z_tensor.clone()
    original_head = z_tensor[:, :, 0, :].clone()
    head_mean_ablation_hook(z_copy, mock_hook, 0)
    expected_mean = original_head.mean(0)
    print(f"✓ Mean ablation - head 0 matches mean: {t.allclose(z_copy[:, :, 0, :], expected_mean)}")


def example_logit_attribution():
    """Demonstrate logit attribution function."""
    print("\n📊 Logit Attribution Example")
    print("=" * 50)
    
    # Create test data
    seq_len, d_model, n_heads, d_vocab = 6, 64, 4, 100
    
    embed = t.randn(seq_len, d_model)
    l1_results = t.randn(seq_len, n_heads, d_model)
    l2_results = t.randn(seq_len, n_heads, d_model)
    W_U = t.randn(d_model, d_vocab)
    tokens = t.randint(0, d_vocab, (seq_len,))
    
    print(f"✓ Created test data:")
    print(f"  Embed: {embed.shape}")
    print(f"  L1 results: {l1_results.shape}")
    print(f"  L2 results: {l2_results.shape}")
    print(f"  W_U: {W_U.shape}")
    print(f"  Tokens: {tokens.shape}")
    
    # Calculate logit attribution
    attribution = logit_attribution(embed, l1_results, l2_results, W_U, tokens)
    print(f"✓ Logit attribution result: {attribution.shape}")
    print(f"  Expected shape: ({seq_len-1}, {1 + 2*n_heads})")
    print(f"  All values finite: {t.all(t.isfinite(attribution))}")


def example_model_loading():
    """Demonstrate model loading and basic usage."""
    print("\n🤖 Model Loading Example")
    print("=" * 50)
    
    try:
        # Load a small model
        print("Loading Pythia 70M model...")
        model = load_model("pythia-70m")
        print(f"✓ Model loaded: {model.cfg.model_name}")
        print(f"  Layers: {model.cfg.n_layers}")
        print(f"  Heads: {model.cfg.n_heads}")
        print(f"  Model dimension: {model.cfg.d_model}")
        print(f"  Device: {model.cfg.device}")
        
        # Create induction score store for this model
        score_store = get_induction_score_store(
            model.cfg.n_layers, 
            model.cfg.n_heads, 
            str(model.cfg.device)
        )
        print(f"✓ Created induction score store: {score_store.shape}")
        
        # Test tokenization
        test_text = "Hello world!"
        tokens = model.to_tokens(test_text)
        print(f"✓ Tokenized '{test_text}': {tokens.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def example_attention_visualization():
    """Demonstrate attention visualization functions."""
    print("\n👁️ Attention Visualization Example")
    print("=" * 50)
    
    # This would require a real model and cache
    print("Note: This example requires a loaded model and activation cache.")
    print("In practice, you would:")
    print("1. Load a model: model = load_model('pythia-70m')")
    print("2. Run with cache: logits, cache = model.run_with_cache('Hello world!')")
    print("3. Visualize attention: attention_head_display(model, 'Hello world!', cache)")
    print("4. Visualize patterns: attention_pattern_display(model, 'Hello world!', cache)")


def main():
    """Main example function."""
    print("🧪 Interpretability Functions Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_functions()
    example_logit_attribution()
    model = example_model_loading()
    example_attention_visualization()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("\n📝 Next steps:")
    print("1. Load a model: model = load_model('pythia-70m')")
    print("2. Run experiments: logits, cache = model.run_with_cache('your text')")
    print("3. Apply hooks: model.run_with_hooks(tokens, fwd_hooks=[...])")
    print("4. Analyze results using the interpretability functions")
    
    if model is not None:
        print(f"\n🎯 Ready to experiment with {model.cfg.model_name}!")


if __name__ == "__main__":
    main()
