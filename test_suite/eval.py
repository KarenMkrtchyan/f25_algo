#%%
from huggingface_hub import login

from lm_eval import evaluator, tasks
import pandas as pd
import torch
import os
from dotenv import load_dotenv

load_dotenv()
hf_api = os.getenv("HUGGINGFACE_KEY")
login(token=hf_api)

# run yaml_generator.py before
#%%

torch.cuda.empty_cache()

MODELS = [
    #"Qwen/Qwen2.5-0.5B",
    #"Qwen/Qwen2.5-1.5B",
    #"Qwen/Qwen2.5-3B",
    #"Qwen/Qwen2.5-7B"

    #"meta-llama/Llama-3.2-1B",
    #"meta-llama/Llama-3.2-3B",
    
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-2-2b",
    #"google/gemma-2-9b",
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
            num_fewshot=10,
            limit=1000,
            task_manager=tm,
            device=device,
    )
    accuracy = res["results"]["greater_than"]["acc,none"] * 100
    results.append({"model_name": model_name, "accuracy": accuracy})

#%%
df = pd.DataFrame(results)

model_count=len(MODELS)
ex=1000
run=1
file_name=f"dataruns/benchmarks/accuracy_eval_gemma_{model_count}models_{ex}examples_10shot_RUN{run}.csv"
df.to_csv(file_name, index=False)

print(f"Results saved in {file_name}")

# %%
