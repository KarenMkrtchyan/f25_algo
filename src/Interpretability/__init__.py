"""
Interpretability package for transformer analysis.
This package contains functions and utilities for mechanistic interpretability experiments.
"""

# Import interpretability functions
from .Functions import (
    hook_function,
    get_induction_score_store,
    induction_score_hook,
    find_induction_heads,
    visualize_pattern_hook,
    logit_attribution,
    plot_logit_attributions,
    head_zero_ablation_hook,
    head_mean_ablation_hook,
    get_ablation_scores,
    visualize_ablation,
    run_model_with_induction_analysis,
    run_model_with_ablation_analysis,
    display_attention_patterns,
    display_attention_heads,
    visualize_neuron_activation,
    get_top_ablated_heads,
    get_top_attention_heads,
    attach_head_ablation_hooks,
    attention_pattern,
    concatenated_attention_patterns,
    concatenated_ablation_patterns,
    logit_lens_df,
    track_tokens_df,
    plot_token_logits,
    test_logit_lens
)

# Version information
__version__ = "1.0.0"

# Package metadata
__author__ = "Your Name"
__description__ = "Interpretability functions for transformer analysis"

# Make functions available at package level
__all__ = [
    "hook_function",
    "get_induction_score_store", 
    "induction_score_hook",
    "find_induction_heads",
    "visualize_pattern_hook",
    "logit_attribution",
    "plot_logit_attributions",
    "head_zero_ablation_hook",
    "head_mean_ablation_hook", 
    "get_ablation_scores",
    "visualize_ablation",
    "run_model_with_induction_analysis",
    "run_model_with_ablation_analysis"
]
