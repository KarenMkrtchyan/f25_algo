# -1 means start layer is TBD and is skipped in the pipeline
modelslist = {
    "llama-3-8b": {'name': "meta-llama/Meta-Llama-3-8B", 'start_layer': 15},
    "qwen2.5-7b": {'name': "Qwen/Qwen2.5-7B", 'start_layer': -1},
    "gemma-7b": {'name':  "google/gemma-7b", 'start_layer': -1},
}