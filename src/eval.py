import torch
from utils.models import *
from utils.icl_prompts import *
from utils.data.dataset_utils import *
from utils.data.data_generator import *
from utils.models import PythiaModel

if torch.cuda.is_available():
    print("cuda is available")
else:
    print("cuda is not available")

torch.cuda.empty_cache()

model = PythiaModel(
    model_name="eleutherai/pythia-70m-deduped",
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step3000"
)

gen = DataGenerator(seed=42)
data1 = gen.synthetic_data_one(1000)

icl = ICLPrompt()

results = []

for i, sample in enumerate(data1):
    prompt = icl.build_prompt(sample, num_examples=5)
    output = model.generate(prompt)

    sample["prompt"] = prompt
    sample["model_output"] = output
    results.append(sample)

write_parquet_shards(results, "../output", "predictions")
print(f"Saved {len(results)} results!")