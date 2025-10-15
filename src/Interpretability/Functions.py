import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import Setup
from utils.model_config import load_model
from utils.Setup import Float, Int, Tensor, HookPoint, t, einops, cv, display, HookedTransformer, Callable, functools, tqdm, utils

def hook_function(
    model: HookedTransformer,
    attn_pattern: Float[Tensor, "batch heads seq_len seq_len"],
    hook: HookPoint
) -> Float[Tensor, "batch heads seq_len seq_len"]:
    """
    Basic hook function that returns the input pattern unchanged.
    
    Args:
        model: The HookedTransformer model
        attn_pattern: Attention pattern tensor
        hook: Hook point object
        
    Returns:
        The input attention pattern unchanged
    """
    return attn_pattern

def get_induction_score_store(model: HookedTransformer) -> t.Tensor:
    """
    Create a tensor to store induction scores for the given model.
    
    Args:
        model: The HookedTransformer model
        
    Returns:
        Zero-initialized tensor for storing induction scores
    """
    return t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

def induction_score_hook(
    model: HookedTransformer,
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"], 
    hook: HookPoint,
    induction_score_store: t.Tensor
) -> None:
    """
    Hook function to calculate induction scores.
    
    Args:
        model: The HookedTransformer model
        pattern: Attention pattern tensor
        hook: Hook point object
        induction_score_store: Tensor to store the scores
    """
    seq_len = pattern.shape[-1]
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    induction_score_store[hook.layer(), :] = induction_score

def find_induction_heads(
    model: HookedTransformer,
    text: str,
    seq_len: int = 50,
    batch_size: int = 10
) -> t.Tensor:
    """
    Find induction heads in the model using repeated tokens.
    
    Args:
        model: The HookedTransformer model
        text: Text to create repeated tokens from
        seq_len: Length of repeated sequence
        batch_size: Number of sequences in batch
        
    Returns:
        Tensor containing induction scores for each head
    """
    # Generate repeated tokens
    tokens = model.to_tokens(text)
    rep_tokens = t.cat([tokens, tokens], dim=1)
    rep_tokens_batch = rep_tokens.repeat(batch_size, 1)
    
    # Create induction score store
    induction_score_store = get_induction_score_store(model)
    
    # Create hook function
    def hook_fn(pattern, hook):
        induction_score_hook(model, pattern, hook, induction_score_store)
    
    # Run model with hooks
    model.run_with_hooks(
        rep_tokens_batch,
        return_type=None,
        fwd_hooks=[(utils.get_act_name("pattern", layer), hook_fn) for layer in range(model.cfg.n_layers)]
    )
    
    return induction_score_store

def visualize_pattern_hook(
    model: HookedTransformer,
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
    tokens: list = None
) -> None:
    """
    Hook function to visualize attention patterns.
    
    Args:
        model: The HookedTransformer model
        pattern: Attention pattern tensor
        hook: Hook point object
        tokens: List of tokens for visualization
    """
    print(f"Layer: {hook.layer()}")
    
    if tokens is None:
        tokens = model.to_str_tokens("Hello world!")
    
    display(
        cv.attention.attention_patterns(
            tokens=tokens, 
            attention=pattern.mean(0)
        )
    )

def logit_attribution(
    model: HookedTransformer,
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Calculate logit attribution for the model.
    
    Args:
        model: The HookedTransformer model
        embed: Token embeddings
        l1_results: Layer 1 attention head outputs
        l2_results: Layer 2 attention head outputs
        tokens: Token IDs
        
    Returns:
        Tensor of logit attributions
    """
    W_U = model.W_U
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(
        W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    l2_attributions = einops.einsum(
        W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

def plot_logit_attributions(model: HookedTransformer, attribution_tensor: t.Tensor) -> None:
    """
    Plot logit attributions.
    
    Args:
        model: The HookedTransformer model
        attribution_tensor: Tensor of logit attributions
    """
    # This is a placeholder - you can implement actual plotting here
    print(f"Logit attributions shape: {attribution_tensor.shape}")
    print(f"Model: {model.cfg.model_name}")

def head_zero_ablation_hook(
    model: HookedTransformer,
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    """
    Hook function to zero out a specific attention head.
    
    Args:
        model: The HookedTransformer model
        z: Attention head outputs
        hook: Hook point object
        head_index_to_ablate: Index of head to ablate
    """
    z[:, :, head_index_to_ablate, :] = 0.0

def head_mean_ablation_hook(
    model: HookedTransformer,
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    """
    Hook function to replace a specific attention head with its mean.
    
    Args:
        model: The HookedTransformer model
        z: Attention head outputs
        hook: Hook point object
        head_index_to_ablate: Index of head to ablate
    """
    z[:, :, head_index_to_ablate, :] = z[:, :, head_index_to_ablate, :].mean(0)

def get_ablation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Calculate ablation scores for all heads in the model.
    
    Args:
        model: The HookedTransformer model
        tokens: Input tokens
        ablation_function: Function to use for ablation
        
    Returns:
        Tensor of ablation scores for each head
    """
    # Initialize ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculate baseline loss
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -t.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1).mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Create hook function
            temp_hook_fn = functools.partial(ablation_function, model, head_index_to_ablate=head)
            
            # Run model with ablation hook
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)]
            )
            
            # Calculate loss difference
            loss = -t.log_softmax(ablated_logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1).mean()
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores

def visualize_ablation(
    model: HookedTransformer,
    ablation_scores: t.Tensor
) -> None:
    """
    Visualize ablation scores.
    
    Args:
        model: The HookedTransformer model
        ablation_scores: Tensor of ablation scores
    """
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Heatmap(
            z=ablation_scores.cpu().numpy(),
            x=list(range(model.cfg.n_heads)),
            y=list(range(model.cfg.n_layers)),
            colorscale='RdBu',
            colorbar=dict(title="Loss Difference")
        ))
        
        fig.update_layout(
            title=f"Ablation Scores for {model.cfg.model_name}",
            xaxis_title="Head",
            yaxis_title="Layer",
            width=900,
            height=350
        )
        
        fig.show()
        
    except ImportError:
        print("Plotly not available. Ablation scores shape:", ablation_scores.shape)
        print("Scores:", ablation_scores)

def run_model_with_induction_analysis(
    model: HookedTransformer,
    text: str,
    seq_len: int = 50,
    batch_size: int = 10
) -> dict:
    """
    Run a complete induction analysis on the model.
    
    Args:
        model: The HookedTransformer model
        text: Text to analyze
        seq_len: Length of repeated sequence
        batch_size: Number of sequences in batch
        
    Returns:
        Dictionary containing analysis results
    """
    # Find induction heads
    induction_scores = find_induction_heads(model, text, seq_len, batch_size)
    
    # Find top induction heads
    top_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            score = induction_scores[layer, head].item()
            if score > 0.1:  # Threshold for "strong" induction
                top_heads.append((layer, head, score))
    
    top_heads.sort(key=lambda x: x[2], reverse=True)
    
    return {
        "induction_scores": induction_scores,
        "top_heads": top_heads,
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads
    }

def run_model_with_ablation_analysis(
    model: HookedTransformer,
    text: str,
    ablation_function: Callable = head_zero_ablation_hook
) -> dict:
    """
    Run a complete ablation analysis on the model.
    
    Args:
        model: The HookedTransformer model
        text: Text to analyze
        ablation_function: Function to use for ablation
        
    Returns:
        Dictionary containing analysis results
    """
    # Tokenize text
    tokens = model.to_tokens(text)
    
    # Get ablation scores
    ablation_scores = get_ablation_scores(model, tokens, ablation_function)
    
    # Find most important heads
    important_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            score = ablation_scores[layer, head].item()
            if score > 0.01:  # Threshold for "important" heads
                important_heads.append((layer, head, score))
    
    important_heads.sort(key=lambda x: x[2], reverse=True)
    
    return {
        "ablation_scores": ablation_scores,
        "important_heads": important_heads,
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads
    }

def visualize_neuron_activation(cache, model, str_tokens, max_k: int = 7):
    """
    Visualize neuron activations for all layers of a HookedTransformer model using CircuitsVis.
    """
    activations_list = []
    for layer in range(model.cfg.n_layers):
        if ("post", layer) in cache:
            act = cache["post", layer]
            # Make sure it's 2D (seq, neurons)
            if act.dim() == 1:
                act = act.unsqueeze(-1)
            elif act.dim() > 2:
                act = act.squeeze(0)
            activations_list.append(act)
        else:
            # Fill missing layers with zeros
            seq_len = str_tokens if isinstance(str_tokens, int) else len(str_tokens)
            n_neurons = getattr(model.cfg, "d_model", 512)
            activations_list.append(t.zeros(seq_len, n_neurons))

    neuron_activations_for_all_layers = t.stack(activations_list, dim=1)

    text_vis = cv.activations.text_neuron_activations(
        tokens=str_tokens,
        activations=neuron_activations_for_all_layers
    )

    neuron_activations_for_all_layers_rearranged = (
        einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons")
        .detach().cpu().numpy()
    )

    topk_vis = cv.topk_tokens.topk_tokens(
        tokens=[str_tokens],
        activations=neuron_activations_for_all_layers_rearranged,
        max_k=max_k,
        first_dimension_name="Layer",
        third_dimension_name="Neuron",
        first_dimension_labels=list(range(model.cfg.n_layers))
    )

    return text_vis, topk_vis, neuron_activations_for_all_layers, neuron_activations_for_all_layers_rearranged


def display_attention_heads(model, text_or_tokens, cache, layer=0, position=0):
    """
    Display attention heads for a specific layer and position using CircuitsVis.

    Args:
        model: HookedTransformer model (loaded via load_model()).
        text_or_tokens: Input text (string) or pre-tokenized tensor/list of tokens.
        cache: ActivationCache from transformer_lens run.
        layer: Layer index to visualize.
        position: Token position index to visualize.
    """
    
    if isinstance(text_or_tokens, str):
        tokens = model.to_str_tokens(text_or_tokens)
    else:
        tokens = model.to_str_tokens(text_or_tokens)

    attention = cache["pattern", layer][0].detach().cpu().numpy()  # batch dim → [head, q, k]

    attention_at_position = attention[:, position, :]  # shape: [head, key_pos]

    vis = cv.attention.attention_heads(tokens=tokens, attention=attention_at_position)
    return vis

def display_attention_patterns(model, text_or_tokens, cache, layer=0, position=0):
    """
    Display full attention patterns for all heads in a given layer using CircuitsVis.

    Args:
        model: HookedTransformer model (loaded via load_model()).
        text_or_tokens: Input text (string) or pre-tokenized tensor/list of tokens.
        cache: ActivationCache from transformer_lens run.
        layer: Layer index to visualize.
        position: Token position index to highlight.
    """

    if isinstance(text_or_tokens, str):
        tokens = model.to_str_tokens(text_or_tokens)
    else:
        tokens = model.to_str_tokens(text_or_tokens)

    attention = cache["pattern", layer][0].detach().cpu().numpy()  # shape: [head, query, key]

    num_heads = model.cfg.n_heads
    head_names = [f"L{layer}H{i}" for i in range(num_heads)]

    vis = cv.attention.attention_patterns(
        tokens=tokens,
        attention=attention,
        attention_head_names=head_names,
    )

    return vis

def activation_patch(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    hook_point: str = "blocks.{layer}.hook_resid_pre",
    layer: int | list[int] = 0,
    token_pos: int | None = None,
    return_logits: bool = True,
):
    """
    Perform activation patching on a HookedTransformer model.
    """
    clean_tokens = model.to_tokens(clean_prompt).to(model.cfg.device)
    corrupted_tokens = model.to_tokens(corrupted_prompt).to(model.cfg.device)

    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    def patch_hook(value, hook):
        clean_value = clean_cache[hook.name]
        if clean_value.shape != value.shape:
            raise ValueError(f"Activation shapes differ at {hook.name}: {clean_value.shape} vs {value.shape}")
        if token_pos is not None:
            value[:, token_pos] = clean_value[:, token_pos]
            return value
        return clean_value

    if isinstance(layer, int):
        layers = [layer]
    else:
        layers = layer

    hooks = [(hook_point.format(layer=l), patch_hook) for l in layers]
    result = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)

    return result if return_logits else (result, clean_cache, corrupted_cache)

