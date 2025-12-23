import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
from typing import Dict, List, DefaultDict, Literal, Union
import numpy as np
DigitPosition = Literal[
    "ten_thousands",
    "thousands",
    "hundreds",
    "tens",
    "units"
]


class FisherCalculations:
    """
    Compute Fisher scores for neuron activations in a HookedTransformer model.

    Fisher score for neuron i:
        F_i = Σ_c n_c (μ_{i,c} - μ_i)^2 / Σ_c n_c σ^2_{i,c}

    This class is fully compatible with 5-digit comparison data.
    """

    def __init__(
        self,
        model: HookedTransformer,
        data: DefaultDict[str, DefaultDict[str, List[Dict]]],
        device: str,
        start_layer: int = 0,
        end_layer: int = 20,
        batch_size: int = 64
    ) -> None:
        self.model = model
        self.data = data
        self.device = device
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.batch_size = batch_size

        self.class_stats = {}
        self.global_mean = None
        self.between_var = None

    def calc_in_class_stats(self, digit_position: DigitPosition) -> None:
        """Compute mean/variance for each digit-pair class."""
        self.class_stats = {}

        for pair_class, examples in tqdm(self.data[digit_position].items(), desc=f"Processing {digit_position}"):

            n = len(examples)
            if n < 2:
                continue

            prompts = [ex["text"] for ex in examples]
            all_acts = []

            for i in range(0, n, self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                tokens = self.model.to_tokens(batch_prompts, padding_side="left").to(self.device)

                with torch.no_grad():
                    logits, cache = self.model.run_with_cache(tokens, remove_batch_dim=False)

                layer_acts = []
                for layer in range(self.start_layer, self.end_layer):
                    activations = cache["mlp_out", layer]
                    seq_lens = (tokens != self.model.tokenizer.pad_token_id).sum(dim=1)
                    last_token = activations[torch.arange(len(seq_lens)), seq_lens - 1]
                    layer_acts.append(last_token)

                layer_acts = torch.stack(layer_acts, dim=0)  # [layers, batch, neurons]
                all_acts.append(layer_acts.cpu())

                del logits, cache
                torch.cuda.empty_cache()

            all_acts = torch.cat(all_acts, dim=1)  # [layers, total_examples, neurons]
            mean_vec = all_acts.mean(dim=1)
            var_vec = all_acts.var(dim=1, unbiased=False)

            self.class_stats[pair_class] = {
                "mean": mean_vec,
                "var": var_vec,
                "n": n,
            }

    def calc_global_mean(self) -> None:
        first_mean = next(iter(self.class_stats.values()))["mean"]
        n_layers, n_neurons = first_mean.shape

        weighted = torch.zeros((n_layers, n_neurons))
        total = 0

        for stats in self.class_stats.values():
            weighted += stats["mean"] * stats["n"]
            total += stats["n"]

        self.global_mean = weighted / total

    def calc_between_class_var(self) -> None:
        n_layers, n_neurons = self.global_mean.shape
        result = torch.zeros((n_layers, n_neurons))
        total = 0

        for stats in self.class_stats.values():
            n = stats["n"]
            result += n * (stats["mean"] - self.global_mean) ** 2
            total += n

        self.between_var = result / max(total - 1, 1)

    def calc_fisher(self) -> np.ndarray:
        eps = 1e-12
        n_layers, n_neurons = self.global_mean.shape

        between = torch.zeros((n_layers, n_neurons))
        within = torch.zeros((n_layers, n_neurons))

        for stats in self.class_stats.values():
            n = stats["n"]
            between += n * (stats["mean"] - self.global_mean) ** 2
            within  += n * stats["var"]

        fisher = between / (within + eps)
        return fisher.cpu().numpy()