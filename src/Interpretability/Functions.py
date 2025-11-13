import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import re
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import yaml
import torch as t
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
    device = model.cfg.device
    tokens = tokens.to(device)
    
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    ablation_scores = t.zeros((n_layers, n_heads), device=device)

    model.reset_hooks()
    with t.no_grad():
        logits = model(tokens, return_type="logits")
        loss_fn = t.nn.CrossEntropyLoss()
        loss_no_ablation = loss_fn(logits.view(-1, logits.shape[-1]), tokens.view(-1))

    hook_names = [utils.get_act_name("z", layer) for layer in range(n_layers)]

    for layer in tqdm(range(n_layers), desc="Layers"):
        for head in range(n_heads):
            temp_hook_fn = functools.partial(ablation_function, model, head_index_to_ablate=head)

            with t.no_grad():
                ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_names[layer], temp_hook_fn)])
                loss_ablated = loss_fn(ablated_logits.view(-1, logits.shape[-1]), tokens.view(-1))

            ablation_scores[layer, head] = (loss_ablated - loss_no_ablation).item()

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
    ablation_function: Callable = head_zero_ablation_hook,
    threshold: float = 0.01,
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
    
    important_heads = [
    (layer, head, ablation_scores[layer, head].item())
    for layer in range(model.cfg.n_layers)
    for head in range(model.cfg.n_heads)
    if ablation_scores[layer, head].item() > threshold
    ]
    
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

def attach_head_ablation_hooks(model_name: str, layers: list[int], heads: list[int]):
    """
    Attach hooks to zero out specific heads in specific layers.

    Args:
        model_name: str name of hooked transforemr
        layers (list[int]): List of layer indices to ablate
        heads (list[int]): List of head indices to ablate in each layer
    """

    model = load_model(model_name)
    for layer in layers:
        hook_name = f"blocks.{layer}.attn.hook_z" 
        for head in heads:
            model.add_hook(
                hook_name,
                lambda z, hook, head_index=head: head_zero_ablation_hook(model, z, hook, head_index)
            )

    return model

def get_top_ablated_heads(concat_df: pd.DataFrame, n: int = 10, group_by: str = "mean"):
    """
    Run ablation analysis and return the top n most important heads.

    Args:
        model: HookedTransformer model
        text (str): Prompt text
        threshold (float): Minimum importance value to consider a head
        n (int): Number of top heads to return
        layers_to_ablate (list[int], optional): layers to ablate
        heads_to_ablate (list[int], optional): heads to ablate
    Returns:
        pd.DataFrame: Top heads with columns ["layer", "head", "importance"]
    """
    
    if group_by not in ["mean", "max", "min", "sum"]:
        raise ValueError("group_by must be either 'mean', 'max', 'min', or 'sum'")

    if not all(col in concat_df.columns for col in ["layer", "head", "importance"]):
        raise ValueError("concat_df must contain columns: 'layer', 'head', 'importance'")

    grouped = concat_df.groupby(["layer", "head"], as_index=False).agg({
        "importance": group_by,
    })

    filtered_df = grouped.sort_values("importance", ascending=False)
    top_heads = filtered_df.head(n)
    
    layers_list = top_heads["layer"].tolist()
    heads_list = top_heads["head"].tolist()
    ablation_scores = top_heads["importance"].tolist()

    return top_heads, layers_list, heads_list, ablation_scores
def attention_pattern(model_name, text):

    model = load_model(model_name)

    tokens = model.to_tokens(text)
    tokens_str = model.to_str_tokens(tokens)

    logits, cache = model.run_with_cache(tokens)

    tokens_str = model.to_str_tokens(tokens)
    for i, tok in enumerate(tokens_str):
        print(i, repr(tok))

    def is_relevant(tok: str) -> bool:
        t_clean = tok.lstrip('▁')
        return bool(re.search(r'\d', t_clean)) or t_clean in [" greater", "greater", "greater ", " less", "less ", "less", ">", " >", "> ", "<", " <", "< " "="]

    relevant_tokens = [i for i, tok in enumerate(tokens_str) if is_relevant(tok)]
    print("Relevant token indices:", relevant_tokens)


    all_stats = []
    for layer in range(model.cfg.n_layers):
        key_name = f"blocks.{layer}.attn.hook_pattern"
        attn = cache[key_name][0]

        for head in range(model.cfg.n_heads):
            attn_head = attn[head]
            if len(relevant_tokens) > 0:
                attn_relevant = attn_head[relevant_tokens, :][:, relevant_tokens]
                mean_val = t.mean(attn_relevant).item()
                min_val = t.min(attn_relevant).item()
                max_val = t.max(attn_relevant).item()
            else:
                mean_val = min_val = max_val = None

            all_stats.append({
                "layer": layer,
                "head": head,
                "mean": mean_val,
                "min": min_val,
                "max": max_val
            })
            
    df_stats = pd.DataFrame(all_stats)
    return df_stats

def concatenated_ablation_patterns(
    model_name: str,
    yaml_path: str,
    n_examples: int = 10,
    n_shots: int = 0,
):
    """
    Runs a model through prompts defined in a YAML task file, collecting 
    attention weights for each layer and head.

    Args:
        model_name: str for HookedTransformer Model.
        yaml_path: Path to YAML task file.
        n_examples: Number of examples to evaluate.
        n_shots: Number of few-shot examples (0 for zero-shot).

    Returns:
        concat_df: Combined DataFrame of all examples.
        individual_dfs: List of per-example DataFrames.
    """

    model = load_model(model_name)

    with open(yaml_path, "r") as f:
        task_cfg = yaml.safe_load(f)
    data_path = task_cfg["dataset_kwargs"]["data_files"][0]

    dataset = pq.read_table(data_path).to_pandas()
    dataset = dataset.sample(n=min(n_examples, len(dataset)), random_state=42)

    def build_prompt(row, dataset_subset):
        a, b = row["a"], row["b"]
        base = f"Is {a} > {b}? Answer:"
        if n_shots <= 0:
            return base
        else:
            few = dataset_subset.sample(n=n_shots)
            shots = "\n".join(
                [f"Is {r.a} > {r.b}? Answer: {'Yes' if r.a > r.b else 'No'}"
                 for _, r in few.iterrows()]
            )
            return shots + "\n" + base

    all_results = []
    individual_results = []

    for i, row in dataset.iterrows():
        prompt = build_prompt(row, dataset)
        
        results = run_model_with_ablation_analysis(model, prompt)
        results_df = pd.DataFrame(results["important_heads"], columns=["layer", "head", "importance"])
        results_df["example_id"] = i
        results_df["n_shots"] = n_shots
        results_df["prompt"] = prompt

        individual_results.append(results_df)
        all_results.append(results_df)

    concat_df = pd.concat(all_results, ignore_index=True)
    return concat_df, individual_results

def concatenated_attention_patterns(
    model_name: str,
    yaml_path: str,
    n_examples: int = 10,
    n_shots: int = 0,
):
    """
    Runs a model through prompts defined in a YAML task file, collecting 
    attention weights for each layer and head.

    Args:
        model_name: str for HookedTransformer Model.
        yaml_path: Path to YAML task file.
        n_examples: Number of examples to evaluate.
        n_shots: Number of few-shot examples (0 for zero-shot).

    Returns:
        concat_df: Combined DataFrame of all examples.
        individual_dfs: List of per-example DataFrames.
    """

    model = load_model(model_name)

    with open(yaml_path, "r") as f:
        task_cfg = yaml.safe_load(f)
    data_path = task_cfg["dataset_kwargs"]["data_files"][0]

    dataset = pq.read_table(data_path).to_pandas()
    dataset = dataset.sample(n=min(n_examples, len(dataset)), random_state=42)

    def build_prompt(row, dataset_subset):
        a, b = row["a"], row["b"]
        base = f"Is {a} > {b}? Answer:"
        if n_shots <= 0:
            return base
        else:
            few = dataset_subset.sample(n=n_shots)
            shots = "\n".join(
                [f"Is {r.a} > {r.b}? Answer: {'Yes' if r.a > r.b else 'No'}"
                 for _, r in few.iterrows()]
            )
            return shots + "\n" + base

    all_results = []
    individual_results = []

    for i, row in dataset.iterrows():
        prompt = build_prompt(row, dataset)
        
        df_stats = attention_pattern_toward_each_token(model, prompt)
        df_stats["example_id"] = i
        df_stats["n_shots"] = n_shots
        df_stats["prompt"] = prompt

        individual_results.append(df_stats)
        all_results.append(df_stats)

    concat_df = pd.concat(all_results, ignore_index=True)
    return concat_df, individual_results

def get_top_attention_heads(concat_df: pd.DataFrame, n: int = 10, sort_by: str = "mean_attention_toward", group_by: str = "mean"):
    """
    Get the top-n attention heads based on their mean or max attention values.

    Args:
        concat_df (pd.DataFrame): Concatenated DataFrame with columns ['layer', 'head', 'mean', 'min', 'max'].
        n (int): Number of top heads to select.
        sort_by (str): Metric to rank heads by — either 'mean' or 'max'.

    Returns:
        top_heads_df (pd.DataFrame): Sorted DataFrame of top-n attention heads.
        top_positions (list[tuple[int, int]]): List of (layer, head) positions of top-n heads.
    """

    if sort_by not in ["mean_attention_toward", "max_attention_toward"]:
        raise ValueError("sort_by must be either 'mean' or 'max'")
    
    if group_by not in ["mean", "max", "min", "sum"]:
        raise ValueError("group_by must be either 'mean', 'max', 'min', or 'sum'")

    grouped = concat_df.groupby(["layer", "head"], as_index=False).agg({
        sort_by: group_by,
    })

    top_heads_df = grouped.sort_values(by=sort_by, ascending=False).head(n).reset_index(drop=True)

    top_layers = top_heads_df["layer"].tolist()
    top_heads = top_heads_df["head"].tolist()
    attention_scores = top_heads_df[sort_by].tolist()

    return top_heads_df, top_layers, top_heads, attention_scores

def attention_pattern_toward_each_token(model, text):
    tokens = model.to_tokens(text)
    tokens_str = model.to_str_tokens(tokens)
    logits, cache = model.run_with_cache(tokens)

    print("Token indices:")
    for i, tok in enumerate(tokens_str):
        print(i, repr(tok))

    def is_relevant(tok: str) -> bool:
        t_clean = tok.lstrip('▁')
        return bool(re.search(r'\d', t_clean)) or t_clean in [" greater", "greater", "greater ", " less", "less ", "less", ">", " >", "> ", "<", " <", "< " "="]

    relevant_indices = [i for i, tok in enumerate(tokens_str) if is_relevant(tok)]
    print("Relevant token indices:", relevant_indices)

    all_stats = []

    for layer in range(model.cfg.n_layers):
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]

        for head in range(model.cfg.n_heads):
            attn_head = attn[head]

            for target_idx in relevant_indices:
                mean_to_token = attn_head[:, target_idx].mean().item()
                max_to_token = attn_head[:, target_idx].max().item()

                all_stats.append({
                    "layer": layer,
                    "head": head,
                    "target_index": target_idx,
                    "target_token": tokens_str[target_idx],
                    "mean_attention_toward": mean_to_token,
                    "max_attention_toward": max_to_token,
                })

    df = pd.DataFrame(all_stats)
    return df

def logit_lens_df(model, text, k=5):
    """
    Runs a logit lens analysis on a TransformerLens HookedTransformer model
    and returns a DataFrame showing the top-k predicted tokens per layer.
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)

    records = []

    for layer_idx in range(model.cfg.n_layers):
        resid = cache["resid_post", layer_idx][0, -1, :]
        pseudo_logits = resid @ model.W_U + model.b_U
        top_vals, top_ids = t.topk(pseudo_logits, k=k)

        for rank, (val, idx) in enumerate(zip(top_vals, top_ids), start=1):
            token_str = repr(model.to_string(idx.unsqueeze(0))).strip("'")
            records.append({
                "layer": layer_idx,
                "rank": rank,
                "token": token_str,
                "logit": val.item()
            })

    return pd.DataFrame(records)

def track_tokens_df(model, text, target_tokens, token_position=-1):
    """
    Tracks the pseudo-logits of multiple candidate tokens across all layers 
    and returns a DataFrame for later analysis/plotting.

    Args:
        model: HookedTransformer (TransformerLens)
        text: Input prompt
        target_tokens: list of candidate strings
        token_position: int, token index to track (-1 = last token)

    Returns:
        pd.DataFrame with columns [layer, token, logit]
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    token_ids_dict = {t: model.to_tokens(t)[0] for t in target_tokens}

    records = []
    
    for layer_idx in range(model.cfg.n_layers):
        resid = cache["resid_post", layer_idx][0, token_position, :]
        pseudo_logits = resid @ model.W_U + model.b_U

        for token_str, ids in token_ids_dict.items():
            logit_val = pseudo_logits[ids].sum().item() if len(ids) > 1 else pseudo_logits[ids[0]].item()
            records.append({
                "layer": layer_idx,
                "token": token_str,
                "logit": logit_val
            })

    df_tokens = pd.DataFrame(records)
    return df_tokens

def plot_token_logits(df_tokens, path=""):
    """
    Plots candidate token pseudo-logits across layers.

    Args:
        df_tokens: DataFrame from track_tokens_df with columns [layer, token, logit]
        path: str, if provided saves the plot to this path
    """
    import matplotlib.pyplot as plt

    candidates = df_tokens['token'].unique()
    plt.figure(figsize=(10,5))

    # Plot each candidate
    for token in candidates:
        token_df = df_tokens[df_tokens['token'] == token]
        plt.plot(token_df['layer'], token_df['logit'], marker='o', label=token)

    # Top token per layer
    top_tokens_per_layer = df_tokens.loc[df_tokens.groupby('layer')['logit'].idxmax()]
    for _, row in top_tokens_per_layer.iterrows():
        plt.scatter(row['layer'], row['logit'], s=100, edgecolor='k', facecolor='none', linewidth=1.5)

    # Final layer top token (safe idxmax within layer subset)
    final_layer = df_tokens['layer'].max()
    layer_subset = df_tokens[df_tokens['layer'] == final_layer]
    final_top_token_row = layer_subset.loc[layer_subset['logit'].idxmax()]
    plt.scatter(final_top_token_row['layer'], final_top_token_row['logit'],
                s=150, color='red', marker='*', label='Final Top Token')

    plt.xlabel("Layer")
    plt.ylabel("Pseudo-logit")
    plt.title("Logit Lens: Candidate Tokens Across Layers")
    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(path, bbox_inches='tight')
        print(f"Plot saved to {path}")
    plt.close()

def test_logit_lens(df_tokens, candidates):
    """
    Sanity checks for logit lens outputs restricted to candidate tokens.

    Args:
        df_tokens: DataFrame from track_tokens_df with columns [layer, token, logit]
        candidates: list of target tokens
    """
    n_layers = df_tokens['layer'].nunique()
    n_tokens = len(candidates)

    # 1️⃣ Check DataFrame shape
    expected_rows = n_layers * n_tokens
    assert df_tokens.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df_tokens.shape[0]}"

    # 2️⃣ Check all candidate tokens are present
    df_tokens_set = set(df_tokens['token'].unique())
    assert df_tokens_set == set(candidates), f"Tokens mismatch. Found: {df_tokens_set}"

    # 3️⃣ Check pseudo-logits vary across layers (not all identical)
    for token in candidates:
        token_logits = df_tokens[df_tokens['token']==token]['logit'].values
        assert len(set(token_logits)) > 1, f"Pseudo-logits do not vary for token '{token}'"

    # 4️⃣ Check top token per layer within candidate subset
    for layer in range(n_layers):
        layer_df = df_tokens[df_tokens['layer']==layer]
        top_token = layer_df.sort_values('logit', ascending=False).iloc[0]['token']
        top_logit = layer_df['logit'].max()
        print(f"Layer {layer}: top candidate token = '{top_token}' (logit={top_logit:.3f})")

    print("✅ All logit lens sanity checks passed for candidate tokens.")


