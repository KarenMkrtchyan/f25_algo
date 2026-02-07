# Understanding universality across different families of LLMs by exploring internal circuits

This is the code for an upcoming paper about the universality theorem examed under mechanistic interpretability

## Setup

### Conda env

```
conda create --name env python=3.10
conda activate env
```

### Install reqs

```
pip install -r requirements.txt
```

### Run

Project in active development, don't run me yet

## Project Structure

```
project/
├── output/                   # Raw dataset files (.parquet)
├── outputs_steering/         # Results and plots from activation steering experiments
├── Results/                  # Comprehensive experimental results (Ablation, Attention, Logits)
├── src/
│   ├── Experiments/          # Core experiment implementations (Ablation, Patching, Logit Lens)
│   ├── fisher_score/         # Pipeline and utils to calculate fisher scores (neuron relevance)
│   ├── steering/             # Activation steering logic and evaluation scripts
│   ├── Interpretability/     # General interpretability functions and path patching
│   ├── test_suite/           # Evaluation benchmarks and task configurations (YAML)
│   ├── utils/                # Data generation, model loading, and shared utilities
│   ├── lm-eval/              # Integration with language model evaluation frameworks
│   └── Evaluator.py          # Main evaluation entry point
├── tasks/                    # Task-specific configurations
└── requirements.txt          # Project dependencies
```
