#%% 
from lm_eval import evaluator, tasks
import pandas as pd
import torch

MODELS = [
    "gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    ]

tm = tasks.TaskManager(include_path=".")

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

results = []

for model_name in MODELS:
    res = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name}",
            tasks=["greater_than"],
            num_fewshot=0,
            limit=10,
            task_manager=tm,
            device=device,
    )
    accuracy = res["results"]["greater_than"]["acc,none"] * 100
    results.append({"model_name": model_name, "accuracy": accuracy})

df = pd.DataFrame(results)

model_count=len(MODELS)
ex=10
run=1
file_name=f"dataruns/accuracy_eval_{model_count}models_{ex}examples_RUN{run}.csv"
df.to_csv(file_name, index=False)

print(f"Results saved in {file_name}")