"""configuration file"""

import torch


def create_gemma_config(
    # Architecture Dimensions
    vocab_size: int = 262_144,
    context_length: int = 32_768,
    emb_dim: int = 640,
    n_heads: int = 4,
    hidden_dim: int = 2048,
    head_dim: int = 256,
    n_kv_groups: int = 1,
    # Layer & Attention Configuration
    n_layers: int = 18,
    sliding_layers_per_block: int = 5,
    full_layers_per_block: int = 1,
    sliding_window: int = 512,
    # RoPE & Normalization
    rope_local_base: float = 10_000.0,
    rope_base: float = 1_000_000.0,
    qk_norm: bool = True,
    query_pre_attn_scalar: int = 256,
    # System
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Generates a Google Gemma configuration dictionary with customizable layer interleaving"""
    # 1. Calculate Block Logic
    block_size = sliding_layers_per_block + full_layers_per_block

    if n_layers % block_size != 0:
        raise ValueError(
            f"Configuration Error: 'n_layers' ({n_layers}) must be a multiple of the "
            f"combined pattern block size ({block_size}).\n"
            f"Pattern: {sliding_layers_per_block} sliding + {full_layers_per_block} full."
        )

    num_blocks = n_layers // block_size

    # 2. Build the layer_types list
    # Create one block pattern
    single_block = ["sliding_attention"] * sliding_layers_per_block + [
        "full_attention"
    ] * full_layers_per_block

    # Repeat the block to fill n_layers
    layer_types = single_block * num_blocks

    # 3. Construct the Dictionary
    config = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "head_dim": head_dim,
        "qk_norm": qk_norm,
        "n_kv_groups": n_kv_groups,
        "rope_local_base": rope_local_base,
        "rope_base": rope_base,
        "sliding_window": sliding_window,
        "layer_types": layer_types,
        "dtype": dtype,
        "query_pre_attn_scalar": query_pre_attn_scalar,
    }

    return config


GEMMA3_CONFIG_CUSTOM = {
    "vocab_size": 50257,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}

# see https://github.com/google/gemma.cpp/blob/3ed403e28707c0e3eb5f48b0e487b63a446d8e2e/gemma/configs.cc#L402
GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}

if __name__ == "__main__":
    pass
