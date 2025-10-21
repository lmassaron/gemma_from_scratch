"""Rotary Positional Embeddings (RoPE) implementation."""

import torch


def compute_rope_params(
    head_dim: int,
    theta_base: int = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
):
    """
    Computes the rotary positional embedding parameters (cosine and sine).

    Args:
        head_dim (int): The dimension of each attention head.
        theta_base (int, optional): The base for the theta value. Defaults to 10_000.
        context_length (int, optional): The maximum sequence length. Defaults to 4096.
        dtype (torch.dtype, optional): The data type for the parameters. Defaults to torch.float32.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and sine parameters.
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles for the rotary embeddings
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine values
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        cos (torch.Tensor): The precomputed cosine values.
        sin (torch.Tensor): The precomputed sine values.

    Returns:
        torch.Tensor: The tensor with RoPE applied.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split the input tensor into two halves
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes for broadcasting
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)
