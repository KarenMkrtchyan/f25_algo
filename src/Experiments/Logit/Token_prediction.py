import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from transformer_lens import utils as Utils
from utils.model_config import load_model

#model_name = "phi-3"
#model_name = "qwen2.5-3b"
model_name = "gemma-2-9b-it"
model = load_model(model_name)

#example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_prompt = "Is 9876 > 5432? Answer:"
example_answer = " Yes"
print("\n")
print("Prompt tokens:", model.to_tokens(example_prompt, prepend_bos = True))
print("Answer tokens:", model.to_tokens(example_answer, prepend_bos = False))
print("\n")
Utils.test_prompt(example_prompt, example_answer, model, prepend_bos = True)
