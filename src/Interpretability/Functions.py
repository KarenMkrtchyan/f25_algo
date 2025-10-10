#This file contains functions related to hooks
import utils.Setup
from utils.model_config import load_model

def hook_function (
        self, 
        attn_pattern: Float[Tensor, "batch heads seq_len seq_len"],
        hook: HookPoint
    ) -> Float[Tensor, "batch heads seq_len seq_len"]:
        return attn_pattern

def get_induction_score_store(n_layers: int, n_heads: int, device: str = "cpu") -> t.Tensor:
    """
    Create a tensor to store induction scores.
    
    Args:
        n_layers: Number of layers in the model
        n_heads: Number of attention heads per layer
        device: Device to create the tensor on
        
    Returns:
        Zero-initialized tensor for storing induction scores
    """
    return t.zeros((n_layers, n_heads), device=device)

def induction_score_hook(pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint):
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    induction_score_store[hook.layer(), :] = induction_score

def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)
        )
    )

    seq_len = 50
    batch_size = 10
    rep_tokens_batch = generate_repeated_tokens(gpt2_small, seq_len, batch_size)

    induction_score_store = t.zeros(
        (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
    )

    gpt2_small.run_with_hooks(
        rep_tokens_batch,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
    )

    imshow(
        induction_score_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".1f",
        width=700,
        height=500,
    )

    # Observation: heads 5.1, 5.5, 6.9, 7.2, 7.10 are all strongly induction-y.
    # Confirm observation by visualizing attn patterns for layers 5 through 7:

    induction_head_layers = [5, 6, 7]
    fwd_hooks = [
        (utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
        for induction_head_layer in induction_head_layers
    ]
    gpt2_small.run_with_hooks(
        rep_tokens,
        return_type=None,
        fwd_hooks=fwd_hooks,
    )

def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(
        W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    l2_attributions = einops.einsum(
        W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

def plot_logit_attributions():
    pass

def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0

def head_mean_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = z[:, :, head_index_to_ablate, :].mean(0)

def get_ablation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
    from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)]
            )
            # Calculate the loss difference (= neg correct logprobs), only on the last seq_len tokens
            loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
            # Store the result, subtracting the clean loss so that a value of 0 means no loss change
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores

def visualize_ablation():
    imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)