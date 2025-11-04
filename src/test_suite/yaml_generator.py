#%%
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                       # repo root
DATA = (ROOT / "output" / "mixed_6_digit-00000.parquet").as_posix()      # correct file

YAML_greater_than_string = f"""
task: mixed_6_digit
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

with open("tasks/mixed_6_digit/mixed_6_digit.yaml", "w") as f:
    f.write(YAML_greater_than_string)

# %%

