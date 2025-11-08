# -1 means start layer is TBD and is skipped in the pipeline
modelslist = [
    {'name': "meta-llama/Meta-Llama-3-8B", 'start_layer': 15},
    {'name': "Qwen/Qwen2.5-7B", 'start_layer': -1},
    {'name':  "google/gemma-7b", 'start_layer': -1},
]