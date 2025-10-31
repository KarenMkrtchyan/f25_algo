import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
from typing import Dict, List, DefaultDict, Literal, Union

DigitPosition = Literal["hundreds", "tens", "units"]

class FisherCalculations:
    """
    Compute Fisher scores for neuron activations in a HookedTransformer model.

    Implements the method described in:
    "Modular Arithmetic: Language Models Solve Math Digit by Digit" (Baeumel et al., 2025).

    Fisher score for neuron i:
        F_i = Σ_c n_c (μ_{i,c} - μ_i)^2 / Σ_c n_c σ^2_{i,c}

    where:
        μ_{i,c}  = mean activation of neuron i for class c
        μ_i      = global mean activation of neuron i
        σ^2_{i,c}= within-class variance of neuron i
        n_c      = number of samples in class c
    """

    def __init__(
        self,
        model: HookedTransformer,
        data: DefaultDict[str, DefaultDict[str, List[Dict]]],
        device: str,
        layer: int = 15,
        batch_size: int = 64
    ) -> None:
        if model.tokenizer is None:
            raise ValueError("Model must have a valid tokenizer. Please ensure the model was loaded correctly.")
    
        self.model = model
        self.data = data
        self.device = device
        self.layer = layer
        self.batch_size = batch_size

        self.class_stats: Dict[str, Dict[str, Union[torch.Tensor, int]]] = {}
        self.global_mean: torch.Tensor | None = None
        self.between_var: torch.Tensor | None = None

    # -------------------------------------------------------------------------
    # Step 1: Compute mean and variance per neuron, per digit-pair class
    # -------------------------------------------------------------------------
    def calc_in_class_stats(self, digit_position: DigitPosition) -> None:
        """
        Compute class-wise mean and variance of neuron activations for a given digit position.
        Saves results in self.class_stats.
        """
        self.class_stats = {}  # reset per run 

        for pair_class, examples in tqdm(self.data[digit_position].items(), desc=f"Processing {digit_position} groups"):
            n = len(examples)
            if n < 2:
                continue  # need at least 2 samples to compute variance

            prompts = [ex["text"] for ex in examples]
            all_acts = []

            # Batched inference
            for i in range(0, n, self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                tokens = self.model.to_tokens(batch_prompts, padding_side="left").to(self.device)

                with torch.no_grad():
                    logits, cache = self.model.run_with_cache(tokens, remove_batch_dim=False)

                activations = cache["mlp_out", self.layer]  # [batch, seq_len, n_neurons]

                # Get final token activations for each sequence
                seq_lens = (tokens != self.model.tokenizer.pad_token_id).sum(dim=1)  # type: ignore
                last_token_acts = activations[
                    torch.arange(len(batch_prompts), device=self.device), seq_lens - 1, :
                ]

                all_acts.append(last_token_acts.detach().cpu())

                # Clean up GPU memory
                del logits, cache, activations, last_token_acts
                torch.cuda.empty_cache()

            # Combine results for this class
            all_acts = torch.cat(all_acts, dim=0)  # [num_examples, n_neurons]
            mean_vec = all_acts.mean(dim=0)
            var_vec = all_acts.var(dim=0, unbiased=False)

            self.class_stats[pair_class] = {
                "mean": mean_vec,
                "var": var_vec,
                "n": n,
            }

    # -------------------------------------------------------------------------
    # Step 2: Compute global mean across all classes
    # -------------------------------------------------------------------------
    def calc_global_mean(self) -> None:
        """
        Compute the global (sample-weighted) mean activation per neuron.
        """
        if not self.class_stats:
            raise ValueError("Run calc_in_class_stats() before computing global mean.")

        n_neurons = next(iter(self.class_stats.values()))["mean"].shape[0]
        device = self.device
        weighted_sum = torch.zeros(n_neurons, device=device)
        total_samples = 0

        for _, stats in self.class_stats.items():
            n = stats["n"]
            weighted_sum += stats["mean"].to(device) * n
            total_samples += n

        self.global_mean = weighted_sum / total_samples

    # -------------------------------------------------------------------------
    # Step 3: Compute between-class variance
    # -------------------------------------------------------------------------
    def calc_between_class_var(self) -> None:
        """
        Compute between-class variance (numerator of Fisher score).
        """
        if self.global_mean is None:
            raise ValueError("Run calc_global_mean() before computing between-class variance.")

        n_neurons = self.global_mean.shape[0]
        device = self.global_mean.device
        numerator_sum = torch.zeros(n_neurons, device=device)
        total_samples = 0

        for _, stats in self.class_stats.items():
            n = stats["n"]
            mean_vec = stats["mean"].to(device)
            numerator_sum += n * (mean_vec - self.global_mean) ** 2
            total_samples += n

        self.between_var = numerator_sum / max(total_samples - 1, 1)

    # -------------------------------------------------------------------------
    # Step 4: Compute Fisher scores for all neurons
    # -------------------------------------------------------------------------
    def calc_fisher(self) -> torch.Tensor:
        """
        Compute Fisher scores for all neurons using previously computed stats.
        Returns a tensor of shape [n_neurons].
        """
        if self.global_mean is None:
            raise ValueError("Run calc_global_mean() before computing Fisher scores.")

        eps = 1e-12
        n_neurons = self.global_mean.shape[0]
        device = self.global_mean.device

        between_sum = torch.zeros(n_neurons, device=device)
        within_sum = torch.zeros(n_neurons, device=device)

        for _, stats in self.class_stats.items():
            n_c = stats["n"]
            mu_ic = stats["mean"].to(device)
            var_ic = stats["var"].to(device)

            between_sum += n_c * (mu_ic - self.global_mean) ** 2
            within_sum += n_c * var_ic

        fisher_scores = between_sum / (within_sum + eps)
        return fisher_scores.cpu()
