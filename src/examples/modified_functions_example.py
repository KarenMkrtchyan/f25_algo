#!/usr/bin/env python3
"""
Example usage of the modified interpretability functions.
This script demonstrates how to use the functions with any HookedTransformer model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import einops

import torch as t
from utils.model_config import load_model
from Interpretability import (
    hook_function,
    get_induction_score_store,
    find_induction_heads,
    logit_attribution,
    get_ablation_scores,
    visualize_ablation,
    run_model_with_induction_analysis,
    run_model_with_ablation_analysis
)


def example_basic_usage():
    """Demonstrate basic usage with any model."""
    print("🔬 Basic Usage Example")
    print("=" * 50)
    
    # Load any model
    model = load_model("pythia-70m")
    print(f"✓ Loaded model: {model.cfg.model_name}")
    
    # Create induction score store for this model
    score_store = get_induction_score_store(model)
    print(f"✓ Created induction score store: {score_store.shape}")
    
    # Test hook function
    mock_hook = type('MockHook', (), {'layer': lambda: 2})()
    attn_pattern = t.randn(1, model.cfg.n_heads, 10, 10)
    result = hook_function(model, attn_pattern, mock_hook)
    print(f"✓ Hook function works: {result.shape}")
    
    print("✅ Basic functions work with any model!")


def example_induction_analysis():
    """Demonstrate induction head analysis."""
    print("\n🔍 Induction Head Analysis Example")
    print("=" * 50)
    
    # Load model
    model = load_model("pythia-70m")
    
    # Run induction analysis
    text = "The cat sat on the mat. The cat sat on the"
    results = run_model_with_induction_analysis(model, text)
    
    print(f"✓ Analyzed model: {results['model_name']}")
    print(f"✓ Found {len(results['top_heads'])} strong induction heads")
    
    if results['top_heads']:
        print("Top induction heads:")
        for layer, head, score in results['top_heads'][:5]:
            print(f"  Layer {layer}, Head {head}: {score:.3f}")
    
    print("✅ Induction analysis complete!")


def example_ablation_analysis():
    """Demonstrate ablation analysis."""
    print("\n⚡ Ablation Analysis Example")
    print("=" * 50)
    
    # Load model
    model = load_model("pythia-70m")
    
    # Run ablation analysis
    text = "The quick brown fox jumps over the lazy dog."
    results = run_model_with_ablation_analysis(model, text)
    
    print(f"✓ Analyzed model: {results['model_name']}")
    print(f"✓ Found {len(results['important_heads'])} important heads")
    
    if results['important_heads']:
        print("Most important heads:")
        for layer, head, score in results['important_heads'][:5]:
            print(f"  Layer {layer}, Head {head}: {score:.3f}")
    
    print("✅ Ablation analysis complete!")


def example_manual_analysis():
    """Demonstrate manual analysis steps."""
    print("\n🛠️ Manual Analysis Example")
    print("=" * 50)
    
    # Load model
    model = load_model("pythia-70m")
    
    # Manual induction head finding
    text = "Hello world! Hello world!"
    induction_scores = find_induction_heads(model, text, seq_len=20, batch_size=5)
    print(f"✓ Manual induction analysis: {induction_scores.shape}")
    
    # Manual ablation analysis
    tokens = model.to_tokens(text)
    ablation_scores = get_ablation_scores(model, tokens)
    print(f"✓ Manual ablation analysis: {ablation_scores.shape}")
    
    # Logit attribution
    logits, cache = model.run_with_cache(tokens)

    l1_z = cache["blocks.0.attn.hook_z"][0] 
    l2_z = cache["blocks.1.attn.hook_z"][0] 
    embed = cache["embed"][0]

    print("Possible W_O attrs:", 
        hasattr(model.blocks[0].attn, "W_O"), 
        hasattr(model.blocks[0].attn, "out_proj") or hasattr(model.blocks[0].attn, "out_proj.weight"))

    W_O = model.blocks[0].attn.W_O          #
    cfg = model.cfg                        
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    d_head = cfg.d_head

    W_O_heads = W_O.view(d_model, n_heads, d_head).permute(1, 2, 0) 

    l1_proj = einops.einsum(l1_z, W_O_heads, "seq nhead d_head, nhead d_head d_model -> seq nhead d_model")
    l2_proj = einops.einsum(l2_z, W_O_heads, "seq nhead d_head, nhead d_head d_model -> seq nhead d_model")

    attribution = logit_attribution(model, embed, l1_proj, l2_proj, tokens[0])

    print(f"✓ Logit attribution: {attribution.shape}")
    
    print("✅ Manual analysis complete!")


def example_different_models():
    """Demonstrate using functions with different models."""
    print("\n🔄 Multiple Models Example")
    print("=" * 50)
    
    models_to_test = ["pythia-70m", "gpt2-small"]
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting {model_name}...")
            model = load_model(model_name)
            
            # Test basic functions
            score_store = get_induction_score_store(model)
            print(f"  ✓ Induction store: {score_store.shape}")
            
            # Test induction analysis
            results = run_model_with_induction_analysis(model, "Test text")
            print(f"  ✓ Induction analysis: {len(results['top_heads'])} strong heads")
            
            print(f"  ✅ {model_name} works perfectly!")
            
        except Exception as e:
            print(f"  ❌ Error with {model_name}: {e}")


def main():
    """Main example function."""
    print("🧪 Modified Interpretability Functions Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_basic_usage()
        example_induction_analysis()
        example_ablation_analysis()
        example_manual_analysis()
        example_different_models()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n📝 Key Benefits:")
        print("• All functions now take model as first parameter")
        print("• Works with any HookedTransformer model")
        print("• Easy to use: function(model, other_params)")
        print("• Consistent API across all functions")
        print("• No hardcoded model dependencies")
        
        print("\n🚀 Usage Pattern:")
        print("```python")
        print("from utils.model_config import load_model")
        print("from Interpretability import find_induction_heads")
        print("")
        print("model = load_model('any-model-name')")
        print("results = find_induction_heads(model, 'your text')")
        print("```")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
