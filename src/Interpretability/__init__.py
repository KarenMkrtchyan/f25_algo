"""
Interpretability package for transformer analysis.
This package contains functions and utilities for mechanistic interpretability experiments.
"""

# Import interpretability functions
from .Functions import (
    hook_function,
    get_induction_score_store,
    induction_score_hook,
    visualize_pattern_hook,
    logit_attribution,
    plot_logit_attributions,
    head_zero_ablation_hook,
    head_mean_ablation_hook,
    get_ablation_scores,
    visualize_ablation
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
    "visualize_pattern_hook",
    "logit_attribution",
    "plot_logit_attributions",
    "head_zero_ablation_hook",
    "head_mean_ablation_hook", 
    "get_ablation_scores",
    "visualize_ablation"
]
