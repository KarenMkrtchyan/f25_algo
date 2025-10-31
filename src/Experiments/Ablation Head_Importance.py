import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import run_model_with_ablation_analysis

model_name = "pythia-160m"
model = load_model(model_name)
text = "Which number is greater: 5 or 3?"
results = run_model_with_ablation_analysis(model, text)

results_folder = "Results"
output_folder = os.path.join(results_folder, "Ablation_Head_Importance")
os.makedirs(output_folder, exist_ok=True)

text = text.replace(" ", "_")
results_df = pd.DataFrame(results["important_heads"], columns=["layer", "head", "importance"])
output_path = os.path.join(output_folder, f"{results['model_name']}--{text}.csv")
results_df.to_csv(output_path, index=False)
print(results_df)