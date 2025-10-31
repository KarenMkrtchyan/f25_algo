#Ignore Pydantic error
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

#Standard libraries
import functools
import sys
from pathlib import Path
from typing import Callable, Dict, Any, Optional

#Installed libraries
import circuitsvis as cv
import einops
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
if MAIN:
    print("Setup complete. Ready for transformer experiments!")

    x = t.randn(2, 3, device=device)
    print("Sample tensor:", x)
