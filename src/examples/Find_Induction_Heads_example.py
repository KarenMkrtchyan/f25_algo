import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import find_induction_heads

model = load_model('pythia-410m')
results = find_induction_heads(model, 'Hello world')
print(results)