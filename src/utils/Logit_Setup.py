from functools import partial
from typing import List, Optional, Union
from .device_utils import get_device

import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch as t
from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
#from IPython.display import HTML, IFrame
from jaxtyping import Float

import transformer_lens.utils as Utils
from transformer_lens import ActivationCache, HookedTransformer

def initialize_environment():
    t.set_grad_enabled(False)
    print("Disabled automatic differentiation")

    device = get_device()

    print(f"\nUsing device: {device}")

    return device