import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch as t
import pandas as pd
from utils import Setup
from Interpretability import attention_pattern

model_name = "pythia-70m"
text = "Is 5 greater than 3"
df_stats = attention_pattern(model_name = model_name, text = text)
print(df_stats)

results_folder = "Results"
output_folder = os.path.join(results_folder, "Attention_Patterns")
os.makedirs(output_folder, exist_ok=True)

text = text.replace(" ", "_")
csv_path = os.path.join(output_folder, f"{model_name}--{text}.csv")
df_stats.to_csv(csv_path, index=False)
