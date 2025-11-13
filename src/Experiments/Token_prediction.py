import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import sys
import os
import torch as t
import transformer_lens.utils as utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup
from utils.model_config import load_model
from transformer_lens import HookedTransformer

model = load_model("gpt2-small")

example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_answer = " Mary"
print("\n")
print("Prompt tokens:", model.to_tokens(example_prompt, prepend_bos = True))
print("Answer tokens:", model.to_tokens(example_answer, prepend_bos = False))
print("\n")
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)