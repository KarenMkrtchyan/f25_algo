from typing import Literal

# -1 means start layer is TBD and is skipped in the pipeline
modelslist = [
    {
        'name': "meta-llama/Meta-Llama-3-8B",
        'start_layer': 15,
        'end_layer': 20
    },
    {
        'name': "Qwen/Qwen2.5-7B",
        'start_layer': -1,
        'end_layer': 20
    },
    {
        'name': "google/gemma-7b",
        'start_layer': -1,
        'end_layer': 20
    }
]

DigitPosition = Literal["hundreds", "tens", "units"]
