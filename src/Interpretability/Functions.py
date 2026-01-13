import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import seaborn as sns
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from tqdm import tqdm
from neel_plotly import imshow, line, scatter
import plotly.subplots as sp
import plotly.graph_objects as go
import yaml
import json
import torch as t
import transformer_lens.patching as patching
from utils import Setup
from functools import partial
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
            color_continuous_scale="RdBu",
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

def ablate_attn_head_last_pos(layer, head):
    def hook(attn_result, hook):
        # attn_result: [batch, pos, head, d_head]
        attn_result[:, -1, head, :] = 0.0
        return attn_result
    return hook

def make_ablated_model(model, layer, head):
    hook = (
        f"blocks.{layer}.attn.hook_result",
        ablate_attn_head_last_pos(layer, head),
    )

    def ablated_forward(tokens, **kwargs):
        with model.hooks(fwd_hooks=[hook]):
            return model(tokens, **kwargs)

    return ablated_forward


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

def get_shared_ylim(df1, df2, pad_frac=0.1):
    lo = min(df1["logit"].min(), df2["logit"].min())
    hi = max(df1["logit"].max(), df2["logit"].max())
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad

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

    attention = cache["pattern", layer]

    attention_matrix = attention[0]

    vis = cv.attention.attention_heads(tokens=tokens, attention=attention_matrix)
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

def make_prompt(a, b):
    return f"Is {a} > {b}? Answer:"

def make_prompt_space(a, b):
    return f"Is {a} > {b}? Answer: "

def build_dataset_space(n=2000, seed=42, low=1, high=100000):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        a = int(rng.integers(low, high))
        b = int(rng.integers(low, a))
        clean = make_prompt_space(a, b)
        corrupt = make_prompt_space(b, a)
        label = int(a > b)
        data.append((clean, corrupt, a, b, label))
    return data

def build_dataset(n=2000, seed=42, low=1, high=100000):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        a = int(rng.integers(low, high))
        b = int(rng.integers(low, a))
        clean = make_prompt(a, b)
        corrupt = make_prompt(b, a)
        label = int(a > b)
        data.append((clean, corrupt, a, b, label))
    return data

def get_last_pos(model, prompt):
    toks = model.to_tokens(prompt)[0]
    return len(toks) - 1 

def logit_diff(logits, index_a, index_b):
    return logits[0, -1, index_a] - logits[0, -1, index_b]

def patch_component(model, corrupt_prompt, clean_cache, hook_name, pos, Yes_index, No_index):
    def hook_fn(corrupt_act, hook):
        clean_act = clean_cache[hook_name]
        if clean_act.ndim == 3:
            new_act = corrupt_act.clone()
            new_act[:, pos, :] = clean_act[:, pos, :].to(corrupt_act.device)
            return new_act
        elif clean_act.ndim == 4:
            new_act = corrupt_act.clone()
            new_act[:, :, pos, :] = clean_act[:, :, pos, :].to(corrupt_act.device)
            return new_act
        else:
            raise ValueError("Unexpected activation shape")

    logits = model.run_with_hooks(
        corrupt_prompt,
        return_type="logits",
        fwd_hooks=[(hook_name, hook_fn)]
    )
    return logit_diff(logits, Yes_index, No_index)

def serialize_cache(cache):
    out = {}
    for name, tensor in cache.items():
        if isinstance(tensor, t.Tensor):
            out[name] = tensor.cpu().tolist()
        else:
            out[name] = tensor
    return json.dumps(out)

def dict_to_torch(cache_dict):
    return {k: t.tensor(v) for k, v in cache_dict.items()}

def load_or_generate_parquet(model, dataset, output_path, Yes_index, No_index):
    if not os.path.exists(output_path):
        print(f"Parquet file not found. Generating caches at {output_path} ...")
        
        schema = pa.schema({
            "clean_caches": pa.string(),
            "corrupt_caches": pa.string(),
            "logit_diffs": pa.float32()
        })
        writer = pq.ParquetWriter(output_path, schema)

        for clean, corrupt, a, b, label in tqdm(dataset):
            clean_logits, clean_cache = model.run_with_cache(clean, remove_batch_dim=False)
            corrupt_logits, corrupt_cache = model.run_with_cache(corrupt, remove_batch_dim=False)

            row = {
                "clean_caches": json.dumps({k: v.cpu().tolist() for k, v in clean_cache.items()}),
                "corrupt_caches": json.dumps({k: v.cpu().tolist() for k, v in corrupt_cache.items()}),
                "logit_diffs": float(logit_diff(corrupt_logits, Yes_index, No_index))
            }

            row = {k: [v] for k, v in row.items()}
            table = pa.Table.from_pydict(row, schema=schema)
            writer.write_table(table)

        writer.close()
        print(f"Parquet file generated at {output_path}")
    else:
        print(f"Parquet file already exists at {output_path}. Streaming caches...")

    def cache_stream():
        pf = pq.ParquetFile(output_path)
        for batch in pf.iter_batches(batch_size=1):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield {
                    "clean_cache": dict_to_torch(json.loads(row["clean_caches"])),
                    "corrupt_cache": dict_to_torch(json.loads(row["corrupt_caches"])),
                    "logit_diff": row["logit_diffs"]
                }

    return cache_stream()

def run_patching(dataset, model, cache_stream, Yes_index, No_index, components=None):
    num_layers = model.cfg.n_layers

    if components is None:
        components = {
            "attn_heads": lambda L: f"blocks.{L}.attn.hook_result",
            "mlp": lambda L: f"blocks.{L}.mlp.hook_post",
            "resid_post": lambda L: f"blocks.{L}.hook_resid_post"
        }

    patch_effects = {}
    for name in components:
        device = model.cfg.device

        if name == "attn_heads":
            patch_effects[name] = t.zeros(num_layers, model.cfg.n_heads, device=device).to(device)
        else:
            patch_effects[name] = t.zeros(num_layers, device=device).to(device)

    for (clean, corrupt, a, b, label), cache_row in tqdm(zip(dataset, cache_stream), total=len(dataset)):
        pos = get_last_pos(model, corrupt)
        clean_cache = {k: v.to(model.cfg.device) for k, v in cache_row["clean_cache"].items()}
        corrupt_cache = {k: v.to(model.cfg.device) for k, v in cache_row["corrupt_cache"].items()}
        base_ld = cache_row["logit_diff"]

        for name, hook_fn in components.items():
            for L in range(num_layers):
                hook_name = hook_fn(L)
                patched_ld = patch_component(model, corrupt, clean_cache, hook_name, pos, Yes_index, No_index)
                patch_effects[name][L] += patched_ld - base_ld

    for name in patch_effects:
        patch_effects[name] /= len(dataset)

    return patch_effects

def plot_attention_head_heatmap(patch_effects, output_path="./figures"):
    attn = patch_effects["attn_heads"].cpu().numpy()
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.imshow(attn, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Δ logit diff (patched - corrupted)")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Attention Head Patch Effects")
    plt.savefig(os.path.join(output_path, "attn_head_patch_heatmap.png"), dpi=300)
    plt.close()

def plot_mlp_patch_bar(patch_effects, output_path="./figures"):
    mlp = patch_effects["mlp"].cpu().numpy()
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(mlp)), mlp)
    plt.title("MLP Patch Effects per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Δ logit diff")
    plt.savefig(os.path.join(output_path, "mlp_patch_effects.png"), dpi=300)
    plt.close()

def plot_resid_patch_bar(patch_effects, output_path="./figures"):
    resid = patch_effects["resid_post"].cpu().numpy()
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(resid)), resid)
    plt.title("Residual Stream Patch Effects")
    plt.xlabel("Layer")
    plt.ylabel("Δ logit diff")
    plt.savefig(os.path.join(output_path, "resid_patch_effects.png"), dpi=300)
    plt.close()

def plot_all_patch_effects(patch_effects, output_path="./figures", save_name="patch_summary.png"):
    os.makedirs(output_path, exist_ok=True)

    resid = patch_effects["resid_post"].detach().cpu().numpy()
    mlp = patch_effects["mlp"].detach().cpu().numpy()
    attn_heads = patch_effects["attn_heads"].detach().cpu().numpy()

    num_layers = attn_heads.shape[0]
    num_heads = attn_heads.shape[1]

    fig = plt.figure(figsize=(18, 6), dpi=200)

    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.2, 1.2], wspace=0.4)

    # ----------------- (a) Residual Stream -----------------
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(resid[:, None], cmap="coolwarm", vmin=-np.abs(resid).max(), vmax=np.abs(resid).max(), aspect="auto")
    ax0.set_title("Residual Stream")
    ax0.set_ylabel("Layer")
    ax0.set_xticks([])
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    # ----------------- (b) MLP Output -----------------
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(mlp[:, None], cmap="coolwarm", vmin=-np.abs(mlp).max(), vmax=np.abs(mlp).max(), aspect="auto")
    ax1.set_title("MLP Output")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # ----------------- (c) Attn Head Output -----------------
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(attn_heads, cmap="coolwarm",
                     vmin=-np.abs(attn_heads).max(), vmax=np.abs(attn_heads).max(),
                     aspect="auto")
    ax2.set_title("Attn Head Output")
    ax2.set_xlabel("Head")
    ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    # ----------------- (d) Mean Attn Across Heads -----------------
    ax3 = fig.add_subplot(gs[0, 3])
    attn_mean = attn_heads.mean(axis=1)
    im3 = ax3.imshow(attn_mean[:, None], cmap="coolwarm",
                     vmin=-np.abs(attn_mean).max(), vmax=np.abs(attn_mean).max(),
                     aspect="auto")
    ax3.set_title("Mean Head Effect")
    ax3.set_yticks([])
    ax3.set_xticks([])
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    # Panel labels
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.text(-0.12, 1.05, f"({chr(ord('a') + i)})",
                transform=ax.transAxes, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, save_name))
    plt.close(fig)

def get_logit_diff(logits, answer_token_indices, mean=True):
    answer_token_indices = answer_token_indices.to(logits.device)

    if logits.ndim == 3:
        logits = logits[:, -1, :]  # final token only

    correct = answer_token_indices[:, 0]
    incorrect = answer_token_indices[:, 1]

    diff = logits.gather(1, correct.unsqueeze(1)) - logits.gather(1, incorrect.unsqueeze(1))

    return diff.mean() if mean else diff


def paper_plot(fig, tickangle=60):
    """
    Applies styling to the given plotly figure object targeting paper plot quality.
    """
    fig.update_layout({
        'template': 'plotly_white',
    })
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=tickangle,
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return fig

def compute_act_patching(model: HookedTransformer,
                         metric: callable,
                         yes_id: int,
                         no_id: int,
                         CLEAN_BASELINE: float,
                         CORRUPTED_BASELINE: float,
                         patching_type: str,
                         batches_base_tokens: list,
                         batches_src_tokens: list,
                         batches: int):
    # resid_streams
    # heads_all_pos : attn heads all positions at the same time
    # heads_last_pos: attn heads last position
    # full: (resid streams, attn block outs and mlp outs)
    list_resid_pre_act_patch_results = []
    for batch in range(batches):
        base_tokens = batches_base_tokens[batch]
        src_tokens = batches_src_tokens[batch]
        base_logits, base_cache = model.run_with_cache(base_tokens)
        src_logits, _ = model.run_with_cache(src_tokens)
        
        metric_fn = partial(metric, yes_id=yes_id, no_id=no_id, CLEAN_BASELINE=CLEAN_BASELINE, CORRUPTED_BASELINE=CORRUPTED_BASELINE)
        if patching_type=='resid_streams':
            # resid_pre_act_patch_results -> [n_layers, pos]
            patch_results = patching.get_act_patch_resid_pre(model, src_tokens, base_cache, metric_fn)
        elif patching_type=='heads_all_pos':
            patch_results = patching.get_act_patch_attn_head_out_all_pos(model, src_tokens, base_cache, metric_fn)
        elif patching_type=='heads_last_pos':
            # Activation patching per position
            attn_head_out_per_pos_patch_results = patching.get_act_patch_attn_head_out_by_pos(model, src_tokens, base_cache, metric_fn)
            # Select last position
            patch_results = attn_head_out_per_pos_patch_results[:,-1]
        elif patching_type=='full':
            patch_results = patching.get_act_patch_block_every(model, src_tokens, base_cache, metric_fn)
        
        list_resid_pre_act_patch_results.append(patch_results)

    total_resid_pre_act_patch_results = t.stack(list_resid_pre_act_patch_results, 0).mean(0)

    return total_resid_pre_act_patch_results

def build_numeric_batches(model, dataset, yes_id, no_id, device, batch_size=32):
    base_toks, src_toks = [], []
    correct_ids, wrong_ids = [], []

    for clean, corrupt, a, b, label in dataset:
        base_toks.append(model.to_tokens(clean, prepend_bos=True))
        src_toks.append(model.to_tokens(corrupt, prepend_bos=True))
        
        correct_ids.append(yes_id)
        wrong_ids.append(no_id)

    max_len = max(seq.shape[-1] for seq in base_toks + src_toks)

    def pad(seq):
        pad_amount = max_len - seq.shape[-1]
        if pad_amount > 0:
            pad_token = model.tokenizer.pad_token_id
            pad_tensor = t.full((1, pad_amount), pad_token, device=device)
            return t.cat([seq.to(device), pad_tensor], dim=-1)
        return seq.to(device)

    base_all = t.cat([pad(x) for x in base_toks], dim=0)
    src_all = t.cat([pad(x) for x in src_toks], dim=0)

    correct_ids = t.tensor(correct_ids, device=device)
    wrong_ids = t.tensor(wrong_ids, device=device)

    batches_base, batches_src, batches_ans = [], [], []

    for i in range(0, len(dataset), batch_size):
        bb = base_all[i:i+batch_size]
        sb = src_all[i:i+batch_size]
        ci = correct_ids[i:i+batch_size]
        wi = wrong_ids[i:i+batch_size]

        ans = t.stack([ci, wi], dim=1)
        batches_base.append(bb)
        batches_src.append(sb)
        batches_ans.append(ans)

    return batches_base, batches_src, batches_ans

def compute_baselines(model, batches_base, batches_src, yes_id, no_id):
    """
    Compute baseline logit diff values:
    - Clean baseline: model confidence for Yes on clean prompts.
    - Corrupt baseline: model confidence for Yes on corrupt prompts.
    """

    base_diffs, src_diffs = [], []

    for bb, sb in zip(batches_base, batches_src):
        base_logits = model(bb)
        src_logits = model(sb)

        # Extract logits for Yes and No at the last position
        base_yes = base_logits[:, -1, yes_id]
        base_no  = base_logits[:, -1, no_id]
        src_yes  = src_logits[:, -1, yes_id]
        src_no   = src_logits[:, -1, no_id]

        # Logit difference: Yes - No
        base_diffs.append(base_yes - base_no)
        src_diffs.append(src_yes - src_no)

    CLEAN_BASELINE = t.cat(base_diffs).mean()
    CORRUPTED_BASELINE = t.cat(src_diffs).mean()
    return CLEAN_BASELINE, CORRUPTED_BASELINE

def numeric_metric(logits, yes_id, no_id, CLEAN_BASELINE, CORRUPTED_BASELINE):
    """
    Normalized metric for activation patching:
    - 1.0 → full rescue (equal to clean baseline)
    - 0.0 → no rescue (equal to corrupt baseline)
    - < 0 → anti-causal (worse than corrupt baseline)
    """

    yes_logits = logits[:, -1, yes_id]
    no_logits  = logits[:, -1, no_id]
    ld = yes_logits - no_logits  # logit diff

    normalized = (ld - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE)
    return normalized.mean()


def plot_all_patch_effects_paper(model, patch_resid, patch_attn, patch_mlp, patch_heads, output_folder):
    """
    Paper-style activation patching visualization using neel_plotly.
    Saves:
        - patch_blocks.png (resid+attn+mlp)
        - patch_heads.png (attn heads)
    """
    os.makedirs(output_folder, exist_ok=True)

    num_layers = model.cfg.n_layers
    num_pos = patch_resid.size(1)
    num_heads = patch_heads.size(1)

    # Flip layers so final layers are shown at top
    resid = t.flip(patch_resid, dims=[0])
    attn = t.flip(patch_attn, dims=[0])
    mlp = t.flip(patch_mlp, dims=[0])
    heads = t.flip(patch_heads, dims=[0])

    y_labels = [f"L{L}" for L in range(num_layers - 1, -1, -1)]
    x_pos = [f"pos {i}" for i in range(num_pos)]
    x_heads = [f"H{i}" for i in range(num_heads)]

    # ============ LEFT FIGURE (3 components shared) ============
    stack = t.stack([resid, attn, mlp], dim=0)

    fig_blocks = imshow(
        stack,
        facet_col=0,
        facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
        y=y_labels,
        x=x_pos,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu",
        title="Residual / Attention / MLP Patch Effects",
        return_fig=True
    )
    fig_blocks.update_xaxes(tickangle=45)
    fig_blocks.update_coloraxes(colorbar=dict(title="Δ Logit Diff"))
    fig_blocks.show()

    blocks_path = os.path.join(output_folder, "patch_blocks.png")
    fig_blocks.write_image(blocks_path, scale=3, width=1100, height=600)
    print(f"Saved: {blocks_path}")

    # ============ RIGHT FIGURE (heads) ============
    fig_heads = imshow(
        heads,
        y=y_labels,
        x=x_heads,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu",
        title="Attention Heads Patch Effects (Last Position)",
        return_fig=True
    )
    fig_heads.update_xaxes(tickangle=45)
    fig_heads.update_coloraxes(colorbar=dict(title="Δ Logit Diff"))
    fig_heads.show()

    heads_path = os.path.join(output_folder, "patch_heads.png")
    fig_heads.write_image(heads_path, scale=3, width=700, height=600)
    print(f"Saved: {heads_path}")

def head_mean_ablation_hook_by_pos(
    z: t.Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
    pos_to_ablate: int,
):
    """
    Hook function to replace a specific attention head at a specific position with its mean.
    
    Args:
        z: Attention head outputs
        hook: Hook point object
        head_index_to_ablate: Index of head to ablate
        pos_to_ablate: position to ablate
    """
    baseline = z[:, :, head_index_to_ablate, :].mean(dim=(0, 1))
    z[:, pos_to_ablate, head_index_to_ablate, :] = baseline

    return z
def head_zero_ablation_hook_by_pos(
    z: t.Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
    pos_to_ablate: int,
):
    """
    Hook function to replace a specific attention head at a specific position with its mean.
    
    Args:
        z: Attention head outputs
        hook: Hook point object
        head_index_to_ablate: Index of head to ablate
        pos_to_ablate: position to ablate
    """
    z[:, pos_to_ablate, head_index_to_ablate, :] = 0.0

    return z



def save_sorted_head_importance(patch_results, output_path="head_importance.csv"):
    """
    Takes patch_results from heads_last_pos activation patching:
    - patch_results shape: [n_layers, n_heads]
    - Writes a CSV sorted by importance (logit diff improvement)
    """

    n_layers, n_heads = patch_results.shape
    data = []

    # Gather head name + score pairs
    for layer in range(n_layers):
        for head in range(n_heads):
            score = patch_results[layer, head].item()
            head_name = f"Layer{layer}Head{head}"
            data.append((head_name, score))

    # Create DataFrame and sort by score descending
    df = pd.DataFrame(data, columns=["Head", "LogitDiff"])
    df_sorted = df.sort_values(by="LogitDiff", ascending=False)

    # Save
    df_sorted.to_csv(output_path, index=False)
    print(f"Saved head ranking to {output_path}")

    return df_sorted

def patch_mlp_neurons(model, layer, batches_base, batches_src, 
                       numeric_metric, CLEAN_BASELINE, CORRUPTED_BASELINE, 
                       yes_id, no_id):

    n_neurons = model.cfg.d_mlp 
    neuron_scores = []
    counter = 0

    # Iterate over neurons in this layer
    for neuron_idx in tqdm(range(n_neurons), desc=f"Patching layer {layer}"):
        scores = []

        # Patch each batch individually then mean-aggregate later
        for bb, sb in zip(batches_base, batches_src):

            # Get clean activation cache
            _, clean_cache = model.run_with_cache(bb)

            def hook(activations, hook):
                activations[:, -1, neuron_idx] = clean_cache[hook.name][:, -1, neuron_idx]
                return activations

            # Patch only neuron_idx at this layer’s mlp.hook_post
            logits = model.run_with_hooks(
                sb,
                fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", hook)]
            )

            score = numeric_metric(logits, yes_id, no_id, CLEAN_BASELINE, CORRUPTED_BASELINE)
            scores.append(score.item())

        neuron_scores.append(np.mean(scores))

    return t.tensor(neuron_scores)

def save_sorted_neuron_importance(neuron_scores, layer, output_csv="neuron_importance_layer.csv"):
    data = [(f"Layer{layer}Neuron{i}", neuron_scores[i].item()) 
            for i in range(len(neuron_scores))]
    df = pd.DataFrame(data, columns=["Neuron", "LogitDiff"])
    df_sorted = df.sort_values(by="LogitDiff", ascending=False)
    df_sorted.to_csv(output_csv, index=False)
    print(f"Saved neuron rankings for L{layer} to {output_csv}")
    return df_sorted

def plot_neuron_scores(neuron_scores, layer, output_path):

    plt.figure(figsize=(14, 4))
    x = list(range(len(neuron_scores)))
    y = neuron_scores.cpu().numpy()

    plt.bar(x, y, width=1.0)
    plt.xlabel("Neuron")
    plt.ylabel("Logit Difference")
    plt.title(f"Neuron Importance in Layer {layer}")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.show()

def plot_component_scores(patch_full, model, output_path=None, label="Component Importance"):
    """
    Visualize causal contribution from transformer components:
    X-axis: Embedding, Attn0, MLP0, Attn1, MLP1, ..., AttnN, MLP(N-1)
    Y-axis: Normalized rescue score (logit difference effect)
    
    patch_full: output from compute_act_patching (shape: [batches, layers, components])
    """

    assert patch_full.ndim == 3, f"Expected 3D tensor [batches, layers, components], got {patch_full.shape}"

    print("📊 patch_full raw shape:", patch_full.shape)

    # 1️⃣ Average over batches
    patch_avg = patch_full.mean(dim=0)  # → [layers, components]
    print("📌 After averaging over batches:", patch_avg.shape)

    n_layers = model.cfg.n_layers
    n_components = patch_avg.shape[1]

    # 2️⃣ We only plot:
    # index 1 = attention output
    # index 4 = mlp output
    # (Verified for Pythia architecture)
    assert n_components > 4, "Component count too small — full patching data unexpected."

    attn_scores = patch_avg[:, 1]
    mlp_scores = patch_avg[:, 4]

    # 3️⃣ Construct flattened sequential plot scores
    flat_scores = [0.0]  # Embedding receives no patch → 0 cause
    labels = ["Emb"]

    for layer in range(n_layers):
        labels.append(f"Attn{layer}")
        flat_scores.append(attn_scores[layer].item())

        labels.append(f"MLP{layer}")
        flat_scores.append(mlp_scores[layer].item())

    x = np.arange(len(flat_scores))

    # 4️⃣ Plot
    plt.figure(figsize=(18, 6))
    plt.plot(x, flat_scores, '-o', markersize=7, linewidth=2,
             label=label)

    plt.xticks(x, labels, rotation=75, ha='right')
    plt.ylabel("Logit Difference")
    plt.xlabel("Model Component")
    plt.title("Average Causal Contribution of Transformer Components")
    plt.grid(True, linestyle='--', alpha=0.4)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"💾 Saved component-level plot to {output_path}")

    plt.show()

def plot_component_scores_lastpos(patch_full, model, output_path=None):
    """
    Component plot with:
      - Embedding (layer 0 only)
      - Alternating Attn / MLP per layer
      - Last-position causal effect
    """

    assert patch_full.ndim == 3, "Expected [component, layer, pos]"

    n_layers = model.cfg.n_layers

    # Take last position
    resid_scores = patch_full[0, :, -1]   # [layers]
    attn_scores  = patch_full[1, :, -1]
    mlp_scores   = patch_full[2, :, -1]

    flat_scores = []
    labels = []

    # Embedding (layer 0 only)
    flat_scores.append(resid_scores[0].item())
    labels.append("Emb0")

    # Alternate Attn / MLP
    for layer in range(n_layers):
        flat_scores.append(attn_scores[layer].item())
        labels.append(f"Attn{layer}")

        flat_scores.append(mlp_scores[layer].item())
        labels.append(f"MLP{layer}")

    x = np.arange(len(flat_scores))

    plt.figure(figsize=(18, 5))
    plt.plot(x, flat_scores, "-o", linewidth=2, markersize=5)

    plt.xticks(
        x,
        labels,
        rotation=65,
        ha="right",
        fontsize=9
    )

    plt.ylabel("Δ Logit Difference (last position)")
    plt.xlabel("Model Component")
    plt.title("Causal Contribution of Transformer Components")

    # Reduce clutter
    plt.grid(axis="y", alpha=0.3)
    plt.margins(x=0.01)
    plt.subplots_adjust(bottom=0.32)  # 👈 spacing for tick labels

    # ❌ No legend

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"Saved component plot to {output_path}")

    plt.show()



    
def get_head_output(cache, model, layer, head, pos):
    d_head = model.cfg.d_head
    n_heads = model.cfg.n_heads

    # Best: hook_result is head-separated value output pre-W_O
    key = f"blocks.{layer}.attn.hook_result"
    if key in cache:
        res = cache[key]
        if res.ndim == 4 and res.shape[2] == n_heads:
            return res[:, pos, head, :]   # [batch, d_head]

    # Backup: GPT-2 unmixed (already [batch, seq, head, d_head])
    key = f"blocks.{layer}.attn.h_out"
    if key in cache:
        return cache[key][:, pos, head, :]

    # Last resort: mixed output, unmix via W_O
    for key in (
        f"blocks.{layer}.hook_attn_out",
        f"blocks.{layer}.attn.hook_attn_out",
        f"blocks.{layer}.attn.out"
    ):
        if key in cache:
            attn_out = cache[key][:, pos, :]  # [batch, d_model]
            W_O = model.blocks[layer].attn.W_O  # [d_model, d_head*n_heads]
            start = head * d_head
            end = (head + 1) * d_head
            return attn_out @ W_O[:, start:end]

    raise KeyError(f"No attention value output found for layer {layer}.")

def plot_head_to_neuron_dot_products(
    model,
    batches_base,
    batches_src,
    Lh, #attention layer
    H,  #attention head
    Lm, #MLP layer
    N,  #Neuron index
    title=None,
    save_path=None
):
    """
    Computes dot(head_out_last_pos, MLP_neuron_w_out) and plots group boxplots
    comparing clean vs corrupt datasets.
    """

    # Lists storing dot products across full dataset
    clean_values = []
    corrupt_values = []
    
    # Output weights of the MLP neuron
    W_out_param = model.blocks[Lm].mlp.W_out

    # Case A: W_out is a Parameter (e.g. Pythia)
    if isinstance(W_out_param, t.nn.Parameter) or isinstance(W_out_param, t.Tensor):
        W = W_out_param

    # Case B: W_out is a Linear module (e.g. GPT-2)
    elif hasattr(W_out_param, "weight"):
        W = W_out_param.weight
    else:
        raise ValueError(f"Cannot find W_out weights. Got type: {type(W_out_param)}")

    # Now handle either shape [d_model, d_mlp] or [d_mlp, d_model]
    if W.shape[0] == model.cfg.d_mlp:  # Pythia style: [2048, 512]
        neuron_w_out = W[N, :]         # correct neuron row
    elif W.shape[1] == model.cfg.d_mlp:  # GPT-2 style: [512, 2048]
        neuron_w_out = W[:, N]
    else:
        raise ValueError(f"Unexpected W_out shape: {W.shape}")
    
    last_pos = -1  # final token
    
    for bb, sb in zip(batches_base, batches_src):
        # Forward passes with cache
        _, clean_cache = model.run_with_cache(bb)
        _, corrupt_cache = model.run_with_cache(sb)

        # Head output of chosen head
        clean_head = get_head_output(clean_cache, model, Lh, H, last_pos)
        corrupt_head = get_head_output(corrupt_cache, model, Lh, H, last_pos)

        # Dot products
        clean_values.extend((clean_head @ neuron_w_out).detach().cpu().tolist())
        corrupt_values.extend((corrupt_head @ neuron_w_out).detach().cpu().tolist())
    
    # Build DataFrame for seaborn
    df = pd.DataFrame({
        "Value": clean_values + corrupt_values,
        "Condition": ["Clean (a>b → Yes)"] * len(clean_values) +
                     ["Corrupt (b>a → No)"] * len(corrupt_values)
    })

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Condition", y="Value", palette=["#88c999", "#c17c74"])
    sns.stripplot(data=df, x="Condition", y="Value", color="black", alpha=0.2)

    plt.ylabel(f"Attn(L{Lh},H{H}) · W_out(L{Lm},N{N})")
    plt.xlabel("Condition")
    if title is not None:
        plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()

    return df

def plot_head_PCA(
    model,
    batches_base,
    batches_src,
    layer: int,
    head: int,
    title=None,
    save_path=None,
    max_points=600,
):

    import torch as t
    import numpy as np
    from sklearn.decomposition import PCA
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    clean_outputs = []
    corrupt_outputs = []

    last_pos = -1

    hook_name = f"blocks.{layer}.attn.hook_z"

    with t.no_grad():
        for bb, sb in zip(batches_base, batches_src):

            # ---- clean ----
            _, clean_cache = model.run_with_cache(
                bb,
                names_filter=hook_name
            )

            # shape: [batch, seq, heads, d_head]
            clean_z = clean_cache[hook_name][:, last_pos, head, :]
            clean_outputs.append(clean_z.cpu())

            del clean_cache
            t.cuda.empty_cache()

            # ---- corrupt ----
            _, corrupt_cache = model.run_with_cache(
                sb,
                names_filter=hook_name
            )

            corrupt_z = corrupt_cache[hook_name][:, last_pos, head, :]
            corrupt_outputs.append(corrupt_z.cpu())

            del corrupt_cache
            t.cuda.empty_cache()

            # early stop once we have enough points
            if sum(x.shape[0] for x in clean_outputs) >= max_points:
                break

    # Stack on CPU
    clean_mat = t.cat(clean_outputs, dim=0)[:max_points].numpy()
    corrupt_mat = t.cat(corrupt_outputs, dim=0)[:max_points].numpy()

    X = np.vstack([clean_mat, corrupt_mat])
    y = np.array([1]*len(clean_mat) + [0]*len(corrupt_mat))

    # PCA (CPU)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Class": np.where(y == 1, "Greater (a>b)", "Less (b>a)")
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Class",
        s=40,
        alpha=0.8
    )

    plt.title(title or f"PCA of Layer {layer}, Head {head}")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return df

def plot_head_input_PCA(
    model,
    batches_base,
    batches_src,
    layer: int,
    title=None,
    save_path=None,
    max_points=600,
):
    """
    PCA on the residual stream input to the attention block
    (i.e. what all heads at this layer read from).

    This corresponds to blocks.{layer}.hook_resid_pre
    """

    import torch as t
    import numpy as np
    from sklearn.decomposition import PCA
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    clean_inputs = []
    corrupt_inputs = []

    last_pos = -1
    hook_name = f"blocks.{layer}.hook_resid_pre"

    with t.no_grad():
        for bb, sb in zip(batches_base, batches_src):

            # ---- clean ----
            _, clean_cache = model.run_with_cache(
                bb,
                names_filter=hook_name
            )

            # shape: [batch, seq, d_model]
            clean_resid = clean_cache[hook_name][:, last_pos, :]
            clean_inputs.append(clean_resid.cpu())

            del clean_cache
            t.cuda.empty_cache()

            # ---- corrupt ----
            _, corrupt_cache = model.run_with_cache(
                sb,
                names_filter=hook_name
            )

            corrupt_resid = corrupt_cache[hook_name][:, last_pos, :]
            corrupt_inputs.append(corrupt_resid.cpu())

            del corrupt_cache
            t.cuda.empty_cache()

            # Early stop
            if sum(x.shape[0] for x in clean_inputs) >= max_points:
                break

    # Stack on CPU
    clean_mat = t.cat(clean_inputs, dim=0)[:max_points].numpy()
    corrupt_mat = t.cat(corrupt_inputs, dim=0)[:max_points].numpy()

    X = np.vstack([clean_mat, corrupt_mat])
    y = np.array([1]*len(clean_mat) + [0]*len(corrupt_mat))

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Class": np.where(y == 1, "Greater (a>b)", "Less (b>a)")
    })

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Class",
        s=40,
        alpha=0.8
    )

    plt.title(title or f"PCA of Residual Input (Layer {layer})")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return df


import torch as t
import numpy as np
import matplotlib.pyplot as plt

def average_logit_tracking(
    model,
    dataset,
    target_tokens,
    token_position=-1,
    max_examples=None,
):
    """
    Computes average pseudo-logits across the dataset for clean and corrupt prompts.

    Returns:
        df_clean_avg, df_corrupt_avg
        Each has columns [layer, token, logit]
    """
    clean_dfs = []
    corrupt_dfs = []

    iterator = dataset if max_examples is None else dataset[:max_examples]

    for clean_text, corrupt_text, _, _, _ in tqdm(iterator, desc="Averaging logit lens"):
        df_clean = track_tokens_df(
            model,
            clean_text,
            target_tokens,
            token_position
        )
        df_corrupt = track_tokens_df(
            model,
            corrupt_text,
            target_tokens,
            token_position
        )

        clean_dfs.append(df_clean)
        corrupt_dfs.append(df_corrupt)

    df_clean_avg = (
        pd.concat(clean_dfs)
          .groupby(["layer", "token"], as_index=False)["logit"]
          .mean()
    )

    df_corrupt_avg = (
        pd.concat(corrupt_dfs)
          .groupby(["layer", "token"], as_index=False)["logit"]
          .mean()
    )

    return df_clean_avg, df_corrupt_avg

def plot_activation_steering(
    model,
    batches_base,
    batches_src,
    yes_id,
    no_id,
    layer,
    head,
    alpha,
    device,
    save_path=None,
    pos=-1,
    max_points=2000,
):
    """
    Figure-7-style activation steering using PC1 of attention head outputs.
    Steering is applied in HEAD SPACE via attn.hook_z.
    """

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    hook_z = f"blocks.{layer}.attn.hook_z"

    # ------------------------------------------------------------
    # 1. Collect head activations (CPU, minimal cache)
    # ------------------------------------------------------------
    clean_acts = []
    corrupt_acts = []

    def collect_acts(batches, store):
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)

                _, cache = model.run_with_cache(
                    batch,
                    names_filter=hook_z
                )

                # [batch, seq, heads, d_head] → [batch, d_head]
                z = cache[hook_z][:, pos, head, :].cpu()
                store.append(z)

                del cache
                torch.cuda.empty_cache()

                if sum(x.shape[0] for x in store) >= max_points:
                    break

    collect_acts(batches_base, clean_acts)
    collect_acts(batches_src, corrupt_acts)

    clean_X = torch.cat(clean_acts, dim=0)[:max_points]
    corrupt_X = torch.cat(corrupt_acts, dim=0)[:max_points]

    X = torch.cat([clean_X, corrupt_X], dim=0).numpy()

    # ------------------------------------------------------------
    # 2. PCA on CPU (PC1 in head space)
    # ------------------------------------------------------------
    pca = PCA(n_components=1)
    pca.fit(X)

    pc1 = torch.tensor(
        pca.components_[0],
        device=device,
        dtype=torch.float32
    )
    pc1 = pc1 / pc1.norm()

    # ------------------------------------------------------------
    # 3. Logit difference helper
    # ------------------------------------------------------------
    def logit_diff(logits):
        return (logits[:, -1, yes_id] - logits[:, -1, no_id]).detach().cpu()

    # ------------------------------------------------------------
    # 4. Steering hook (HEAD SPACE)
    # ------------------------------------------------------------
    def make_steering_hook(pc1, alpha, head, pos, sign):
        def hook(z, hook):
            # z: [batch, seq, heads, d_head]
            z[:, pos, head, :] += sign * alpha * pc1
            return z
        return hook

    # ------------------------------------------------------------
    # 5. Run batches with / without steering
    # ------------------------------------------------------------
    def run_batches(batches, steer=False, sign=+1):
        diffs = []

        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)

                if not steer:
                    logits = model(batch)
                else:
                    with model.hooks(
                        fwd_hooks=[
                            (
                                hook_z,
                                make_steering_hook(pc1, alpha, head, pos, sign),
                            )
                        ]
                    ):
                        logits = model(batch)

                diffs.append(logit_diff(logits))

        return torch.cat(diffs).numpy()

    clean_before = run_batches(batches_base, steer=False)
    clean_after  = run_batches(batches_base, steer=True, sign=1)

    corrupt_before = run_batches(batches_src, steer=False)
    corrupt_after  = run_batches(batches_src, steer=True, sign=-1)

    # ------------------------------------------------------------
    # 6. Statistics
    # ------------------------------------------------------------
    means = [
        clean_before.mean(),
        clean_after.mean(),
        corrupt_before.mean(),
        corrupt_after.mean(),
    ]

    sems = [
        clean_before.std() / np.sqrt(len(clean_before)),
        clean_after.std() / np.sqrt(len(clean_after)),
        corrupt_before.std() / np.sqrt(len(corrupt_before)),
        corrupt_after.std() / np.sqrt(len(corrupt_after)),
    ]

    # ------------------------------------------------------------
    # 7. Plot (Figure-7 style)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    axes[0].bar(["Before", "After"], means[:2], yerr=sems[:2], capsize=5)
    axes[0].set_title("a > b")
    axes[0].set_ylabel("Logit Difference (Yes − No)")

    axes[1].bar(["Before", "After"], means[2:], yerr=sems[2:], capsize=5)
    axes[1].set_title("b > a")

    fig.suptitle(
        f"PCA Steering — L{layer}H{head}, α={alpha}, pos={pos}",
        fontsize=12,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return pc1

def collect_final_logits(
    model,
    dataset,
    yes_id,
    no_id,
    token_position=-1,
    max_examples=None,
):
    """
    Collects final-token logits for Yes/No on clean and corrupt prompts.

    Returns:
        clean_yes, clean_no, corrupt_yes, corrupt_no
        (each is a list of floats)
    """

    clean_yes, clean_no = [], []
    corrupt_yes, corrupt_no = [], []

    iterator = dataset if max_examples is None else dataset[:max_examples]

    for clean_text, corrupt_text, *_ in tqdm(
        iterator,
        desc="Collecting final logits",
    ):
        # ---- Clean ----
        tokens = model.to_tokens(clean_text)
        logits = model(tokens)[0, token_position]

        clean_yes.append(logits[yes_id].item())
        clean_no.append(logits[no_id].item())

        # ---- Corrupt ----
        tokens = model.to_tokens(corrupt_text)
        logits = model(tokens)[0, token_position]

        corrupt_yes.append(logits[yes_id].item())
        corrupt_no.append(logits[no_id].item())

    return clean_yes, clean_no, corrupt_yes, corrupt_no

def plot_final_logits_box(
    clean_yes,
    clean_no,
    corrupt_yes,
    corrupt_no,
    save_path=None,
):
    """
    clean_yes, clean_no, corrupt_yes, corrupt_no:
        lists or 1D tensors of final-token logits
    """

    data = [
        clean_yes,
        clean_no,
        corrupt_yes,
        corrupt_no,
    ]

    # Positions control spacing
    positions = [1, 2, 4, 5]  # gap between clean (1,2) and corrupt (4,5)

    fig, ax = plt.subplots(figsize=(8, 5))

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )

    # Color boxes
    colors = [
        "tab:blue",   # Clean Yes
        "tab:blue",   # Clean No
        "orange",     # Corrupt Yes
        "orange",     # Corrupt No
    ]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # X-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Clean Yes", "Clean No", "Corrupt Yes", "Corrupt No"],
        rotation=0,
    )

    ax.set_ylabel("Final Logit Value")
    ax.set_title("Final Token Logits: Clean vs Corrupt")

    # Optional vertical separator
    ax.axvline(3, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()
