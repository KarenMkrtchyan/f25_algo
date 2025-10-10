import circuitsvis as cv # type: ignore

def attention_head_display(model, text, cache, layer=0, position=0):
    """
    Display attention heads for a specific layer and position.
    Args:
        model: The transformer model (should have .to_str_tokens).
        text: The input text (string or tokens).
        cache: ActivationCache from transformer_lens run.
        layer: Layer index to visualize.
        position: Position index to visualize.
    """
    tokens = model.to_str_tokens(text)
    attention = cache["pattern", layer][position]
    vis = cv.attention.attention_heads(tokens=tokens, attention=attention)
    return vis

def attention_pattern_display(model, text, cache, layer=0, position=0, num_heads=12):
    """
    Display attention patterns for a specific layer and position.
    Args:
        model: The transformer model (should have .to_str_tokens).
        text: The input text (string or tokens).
        cache: ActivationCache from transformer_lens run.
        layer: Layer index to visualize.
        position: Position index to visualize.
        num_heads: Number of heads in the layer.
    """
    tokens = model.to_str_tokens(text)
    attention = cache["pattern", layer][position]
    head_names = [f"L{layer}H{i}" for i in range(num_heads)]
    vis = cv.attention.attention_patterns(
        tokens=tokens,
        attention=attention,
        attention_head_names=head_names,
    )
    return vis