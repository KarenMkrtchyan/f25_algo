import torch
from utils.models import *
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
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000"
)
# try to load data instead of gen 
gen = DataGenerator(seed=42) 
data1 = gen.synthetic_data_one(10)

results = []

for i, sample in enumerate(data1):
    prompt = sample["text"]
    output = model.generate(prompt)

    sample["model_output"] = output
    results.append(sample)

write_parquet_shards(results, "../output", "predictions")
print(f"Saved {len(results)} results!")