#%%
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from test_suite.eval import run_benchmark

model_names = [
    #"Qwen/Qwen2.5-7B", 
    "Qwen/Qwen2.5-3B", 
    #"google/gemma-2-2b", 
    #"meta-llama/Llama-3.2-3B",
    ]

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

accuracy_df = pd.DataFrame(columns=['model_name', 'accuracy', 'digits'])

for model in model_names:

    digit_counter = 3
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
        digit_counter += 1


# %%
# Plot accuracy scores

df = pd.read_csv('Results/accuracy/accuracy_scores.csv', skiprows=1, header=None, names=['idx', 'model_name', 'accuracy', 'digits'] ).drop(columns='idx')

plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='digits', y='accuracy', hue='model_name', marker='o')
plt.xlabel('Digits')
plt.ylabel('Accuracy %')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='Random Baseline')
plt.legend(title='Model')

plt.savefig('Results/accuracy/accuracy_plot.png')
plt.show()

# %%
