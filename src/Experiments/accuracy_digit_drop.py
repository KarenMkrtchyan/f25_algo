#%%
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_suite.eval import run_benchmark

model = "Qwen/Qwen2.5-3b"

df = run_benchmark(
    models=[model],
    task_name="greater_than",
    limit=1000,
    output_dir= "../test_suite/dataruns/benchmarks",
    run=1
)


# %%
