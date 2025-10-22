import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import run_model_with_ablation_analysis

model_name = "pythia-2.8b"
model = load_model(model_name)
text = "Which number is greater: 3 or 5?"
results = run_model_with_ablation_analysis(model, text)

output_folder = "Ablation Head Importance"
os.makedirs(output_folder, exist_ok=True)

results_df = pd.DataFrame(results["important_heads"], columns=["layer", "head", "importance"])
output_path = os.path.join(output_folder, f"{results['model_name']}_important_heads.csv")
results_df.to_csv(output_path, index=False)
print(results_df)