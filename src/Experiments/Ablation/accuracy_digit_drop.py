#%%
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from test_suite.eval import run_benchmark
from dotenv import load_dotenv
from huggingface_hub import login

#%%
model_names = [
    "Qwen/Qwen2-1.5B",	
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    # "Qwen/Qwen2.5-7B", 
    # "Qwen/Qwen2.5-3B", 
    # "google/gemma-2-2b", 
    # "meta-llama/Llama-3.2-3B",
    #"microsoft/Phi-3-mini-4k-instruct",
    # "Qwen/Qwen2.5-1.5b",
    # "Qwen/Qwen3-1.7b",
    #"meta-llama/Llama-3.1-8B-Instruct",
    #"google/gemma-7b-it",
    #"google/gemma-2-9b-it",
    #"Qwen/Qwen3-8B"
    ]

tasks=[
       "greater_than_4_digit",
    #    "greater_than_3_digit", 
    #    "greater_than_5_digit",
    #    "greater_than_6_digit",
    #    "greater_than_7_digit",
    #    "greater_than_8_digit",
    #    "greater_than_9_digit",
    #    "greater_than_10_digit",
    #    "greater_than_11_digit",
    #    "greater_than_12_digit",
    #    "greater_than_13_digit",
    #    "greater_than_14_digit",
    #    "greater_than_15_digit",
    #    "greater_than_16_digit",
    #    "greater_than_17_digit",
    #    "greater_than_18_digit",

    ]

accuracy_df = pd.DataFrame(columns=['model_name', 'accuracy', 'digits'])

for model in model_names:

    for task in tasks:

        df = run_benchmark(
            model=model,
            task_name=task,
            num_fewshot=0,
            limit=1000,
            run=1
        )
        #new_row = {'model_name': model, 'accuracy' : [df.iloc[0]['overall_accuracy']], 'digits': [digit_counter]}
        #accuracy_df = pd.concat([accuracy_df, pd.DataFrame(new_row)])

# %%
