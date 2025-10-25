from huggingface_hub import login

login(token="")
from lm_eval import evaluator, tasks
import pandas as pd
import torch

MODELS = [
    # Small models for quick evaluation
    "gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",

    # More GPT-2 variants
    "gpt2-medium",
    "gpt2-large",
    
    # More Pythia models (scaling sizes)
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    
    # OPT models from Meta
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    
    # Other small models
    "distilgpt2",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-1.3B",
    
    # Bloom models
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    
    # Qwen (Alibaba)
    "Qwen/Qwen2-0.5B",              # 0.5B - Very small, fast
    "Qwen/Qwen2-1.5B",              # 1.5B - Good balance
    "Qwen/Qwen2-7B",                # 7B - Strong performance
    "Qwen/Qwen1.5-0.5B",            # 0.5B - Previous gen
    "Qwen/Qwen1.5-1.8B",            # 1.8B - Previous gen
    "Qwen/Qwen1.5-4B",              # 4B - Previous gen
    "Qwen/Qwen1.5-7B",              # 7B - Previous gen

    # Phi (Microsoft)
    "microsoft/phi-2",               # 2.7B - Very capable
    "microsoft/phi-1_5",             # 1.3B - Smaller version

    #Llama 3 (Meta)
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",

    #Google (Gemma)
    "google/gemma-3-4b-it",


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
            num_fewshot=5,
            limit=100,
            task_manager=tm,
            device=device,
    )
    accuracy = res["results"]["greater_than"]["acc,none"] * 100
    results.append({"model_name": model_name, "accuracy": accuracy})

df = pd.DataFrame(results)

model_count=len(MODELS)
ex=10
run=1
file_name=f"dataruns/accuracy_eval_{3}models_{ex}examples_RUN{run}.csv"
df.to_csv(file_name, index=False)

print(f"Results saved in {file_name}")
