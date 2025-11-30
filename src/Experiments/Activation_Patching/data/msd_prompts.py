#%%
import random

def generate_msd_dataset(n_samples=50):
    prompts = []
    random.seed(42)
    
    for _ in range(n_samples):
        msd1 = random.randint(6, 9)
        msd2 = random.randint(1, 4)
        
        lower_digits = random.randint(0, 999)
        
        num1 = msd1 * 1000 + lower_digits
        num2 = msd2 * 1000 + lower_digits
        
        prompts.append({
            "clean_prompt": f"Is {num1} > {num2}? Answer: ",
            "corrupted_prompt": f"Is {num2} > {num1}? Answer: ",
            "clean_label": "Yes",
            "corrupted_label": "No"
        })
    
    return prompts
# %%
