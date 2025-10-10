import utils.Setup

def visualizing_neuron_activation(gpt2_cache, gpt2_small, gpt2_str_tokens, max_k=7):
    """
    Visualize neuron activations for all layers using CircuitsVis.

    Args:
        gpt2_cache: ActivationCache from transformer_lens run.
        gpt2_small: The model object (should have .cfg.n_layers).
        gpt2_str_tokens: List of string tokens.
        max_k: Number of top tokens to show per neuron.
    """
    neuron_activations_for_all_layers = t.stack(
        [gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)], 
        dim=1
    )   
    # shape: (seq_pos, layers, neurons)

    text_vis = cv.activations.text_neuron_activations(
        tokens=gpt2_str_tokens,
        activations=neuron_activations_for_all_layers
    )

    neuron_activations_for_all_layers_rearranged = (
        einops.rearrange(
            neuron_activations_for_all_layers, 
            "seq layers neurons -> 1 layers seq neurons"
        )
        .detach()
        .cpu()
        .numpy()
    )

    topk_vis = cv.topk_tokens.topk_tokens(
        # Some weird indexing required here ¯\_(ツ)_/¯
        tokens=[gpt2_str_tokens],
        activations=neuron_activations_for_all_layers_rearranged,
        max_k=max_k,
        first_dimension_name="Layer",
        third_dimension_name="Neuron",
        first_dimension_labels=list(range(gpt2_small.cfg.n_layers))
    )

    return text_vis, topk_vis, neuron_activations_for_all_layers, neuron_activations_for_all_layers_rearranged