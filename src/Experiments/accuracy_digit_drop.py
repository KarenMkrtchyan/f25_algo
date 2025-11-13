#%%
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_suite.eval import run_benchmark

model = "Qwen/Qwen2.5-7B"

tasks=[
       "greater_than_3_digit", 
       "greater_than_4_digit",
       "greater_than_5_digit",
       "greater_than_6_digit",
       "greater_than_7_digit",
       "greater_than_8_digit",
       "greater_than_9_digit",
       "greater_than_10_digit",
       "greater_than_11_digit",
       "greater_than_12_digit",
       "greater_than_13_digit",
       "greater_than_14_digit",
       "greater_than_15_digit",
       "greater_than_16_digit",
       "greater_than_17_digit",
       "greater_than_18_digit",
       ]

for task in tasks:

    df = run_benchmark(
        models=[model],
        task_name=task,
        num_fewshot=5,
        limit=1000,
        output_dir= "../test_suite/dataruns/benchmarks/by_digit",
        run=1
    )



# %%
