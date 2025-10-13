from .Setup import (
    device,
    HookedTransformer,
    HookedTransformerConfig,
    ActivationCache,
    FactoredMatrix,
    utils as transformer_utils,
    HookPoint,
    Float,
    Int,
    Tensor,
    t,
    cv,
    einops,
    np,
    nn,
    F,
    eindex,
    display,
    tqdm,
    functools,
    Callable,
    Dict,
    Any,
    Optional
)

from .model_config import (
    load_model,
    create_custom_model,
    get_model_config,
    get_model_info,
    get_model_summary,
    list_available_models,
    get_induction_score_store,
    COMMON_MODELS
)

from .models import PythiaModel

# Version information
__version__ = "1.0.0"

__author__ = "Your Name"
__description__ = "Utilities for transformer interpretability experiments"

__all__ = [
    # Setup imports
    "device",
    "HookedTransformer", 
    "HookedTransformerConfig",
    "ActivationCache",
    "FactoredMatrix",
    "transformer_utils",
    "HookedPoint",
    "Float",
    "Int", 
    "Tensor",
    "t",
    "cv",
    "einops",
    "np",
    "nn",
    "F",
    "eindex",
    "display",
    "tqdm",
    "functools",
    "Callable",
    "Dict",
    "Any",
    "Optional",
    
    # Model config imports
    "load_model",
    "create_custom_model", 
    "get_model_config",
    "get_model_info",
    "get_model_summary",
    "list_available_models",
    "get_induction_score_store",
    "COMMON_MODELS",
    
    # Model imports
    "PythiaModel",
]
