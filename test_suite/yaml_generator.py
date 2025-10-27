#%%
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                       # repo root
DATA = (ROOT / "output" / "data1-00000.parquet").as_posix()      # correct file

YAML_greater_than_string = f"""
task: greater_than
dataset_path: parquet
dataset_name: null
dataset_kwargs:
  data_files:
    - "{DATA}"
validation_split: train
output_type: multiple_choice
doc_to_text: "Is {{{{a}}}} > {{{{b}}}}? Answer:"
doc_to_choice: ["Yes", "No"]
doc_to_target: "{{{{ 0 if (a > b) else 1 }}}}"
"""
Path("tasks/greater_than").mkdir(parents=True, exist_ok=True)

with open("tasks/greater_than/greater_than.yaml", "w") as f:
    f.write(YAML_greater_than_string)

# %%
