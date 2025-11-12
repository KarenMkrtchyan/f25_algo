#%%
from huggingface_hub import login

import pandas as pd
import torch
import os
from dotenv import load_dotenv
from lm_eval import evaluator, tasks
from tl_eval.lm_evaluator import *

load_dotenv()
hf_api = os.getenv("HUGGINGFACE_KEY")
login(token=hf_api)

# run yaml_generator.py before

# %%

def run_benchmark(models, task_name, num_fewshot=0, limit=1000, output_dir="dataruns/benchmarks", run=1):
    import os
    torch.cuda.empty_cache()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_dir = os.path.join(current_dir, "../test_suite/tasks")
    tm = tasks.TaskManager(include_path=tasks_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for model_idx, model in enumerate(models):
        if hasattr(model, "tokenizer") and callable(getattr(model, "eval", None)):
            res = evaluate_lm_eval(model, [task_name], batch_size=1, num_fewshot=num_fewshot, limit=limit)
        else:
            res = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model}",
                tasks=[task_name],
                num_fewshot=num_fewshot,
                limit=limit,
                task_manager=tm,
                device=device,
                log_samples=True,
            )
        accuracy = res["results"][task_name]["acc,none"] * 100

        samples = res.get("samples", {}).get(task_name, [])

        for idx, sample in enumerate(samples):
            doc = sample.get("doc", {})
            question_text = doc.get("text", "")
            is_correct = sample.get("acc", 0) == 1.0

            filtered_resps = sample.get("filtered_resps", [])
            if len(filtered_resps) >= 2:
                yes_logprob, yes_chosen = filtered_resps[0]
                no_logprob, no_chosen = filtered_resps[1]
                model_chose = "Yes" if yes_chosen else "No"
            else:
                model_chose = "Unknown"
            
            correct_answer = "Yes" if sample.get("target") == 0 else "No"
            
            results.append({
                "model_name": model if isinstance(model, str) else f"custom_model_{model_idx}",
                "overall_accuracy": accuracy,
                "sample_id": idx,
                "question": question_text,
                "correct_answer": correct_answer,
                "model_answer": model_chose,
                "is_correct": is_correct,
            })

    df = pd.DataFrame(results)
    file_name = f"{output_dir}/qwen2.5-3B/accuracy_eval_qwen2.5-3b_ABLATED_{task_name}_{num_fewshot}shot_RUN{run}.csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, index=False)

    print(f"Results saved in {file_name}")
    return df

# %%
