from huggingface_hub import login

import pandas as pd
import torch
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lm_eval import evaluator, tasks
from tl_eval.lm_evaluator import *

load_dotenv()
hf_api = os.getenv("HUGGINGFACE_KEY")
login(token=hf_api)

def run_benchmark(model, task_name, num_fewshot=0, limit=1000, run=1, ablated_head="",ablated_pos=""):
    torch.cuda.empty_cache()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.abspath(
        os.path.join(current_dir, "dataruns", "benchmarks")
    )

    tasks_dir = os.path.join(current_dir, "../test_suite/tasks")
    tm = tasks.TaskManager(include_path=tasks_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    if hasattr(model, "tokenizer") and callable(getattr(model, "eval", None)):
        res = evaluate_lm_eval(
            model, [task_name],
            batch_size=1, num_fewshot=num_fewshot, limit=limit
        )
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

        a = int(doc["a"])
        b = int(doc["b"])
        question_text = f"Is {a} > {b}? Answer:"

        filtered_resps = sample.get("filtered_resps", [])
        if len(filtered_resps) >= 2:
            yes_vals = filtered_resps[0]
            no_vals  = filtered_resps[1]

            yes_logprob, yes_chosen = yes_vals
            no_logprob,  no_chosen  = no_vals

            if yes_chosen:
                model_chose = "Yes"
            elif no_chosen:
                model_chose = "No"
            else:
                model_chose = "Yes" if yes_logprob > no_logprob else "No"
        else:
            model_chose = "Unknown"

        truth = (a > b)
        correct_answer = "Yes" if truth else "No"
        is_correct = (model_chose == correct_answer)

        results.append({
            "model_name": model if isinstance(model, str) else "custom_model",
            "overall_accuracy": accuracy,
            "sample_id": idx,
            "question": question_text,
            "correct_answer": correct_answer,
            "model_answer": model_chose,
            "is_correct": is_correct,
        })

    df = pd.DataFrame(results)

    model_name_str = model if isinstance(model, str) else "custom_model"
    model_name_str = model_name_str.replace("/", "__")

    file_name = os.path.join(
        output_dir,
        model_name_str,
        f"accuracy_eval_{model_name_str}_{ablated_head}_{ablated_pos}_{task_name}_{num_fewshot}shot_RUN{run}.csv",
    )

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, index=False)

    print(f"Results saved in {file_name}")
    return df
