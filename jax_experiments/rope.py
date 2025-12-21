"""Rotary Positional Embeddings (RoPE) implementation in JAX."""

import jax.numpy as jnp


def compute_rope_params(
    head_dim: int,
    theta_base: int = 10_000,
    context_length: int = 4096,
    dtype: jnp.dtype = jnp.float32,
):
    """
    Computes the rotary positional embedding parameters (cosine and sine).
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 1. Compute the inverse frequencies for the rotations.
    inv_freq = 1.0 / (
        theta_base
        ** (jnp.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)] / head_dim)
    )

    # 2. Generate position indices.
    positions = jnp.arange(context_length, dtype=dtype)

    # 3. Compute the angles.
    angles = positions[:, None] * inv_freq[None, :]

    # 4. Duplicate the angles for each pair of dimensions.
    angles = jnp.concatenate([angles, angles], axis=1)

    # 5. Precompute the sine and cosine values.
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    return cos, sin


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 1. Split the input tensor's last dimension into two halves.
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # 2. Adjust the shapes of sin and cos for broadcasting.
    # Expected x shape: (batch, heads, seq_len, head_dim)
    # cos/sin shape: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)

    # Slice to current sequence length if needed (though usually we pass sliced cos/sin)
    # But for safety in JAX (static shapes usually), we assume inputs are correct size
    # or we handle slicing outside. Here we assume x matches seq_len or we rely on broadcasting.

    # We slice cos/sin to match x's seq_len
    cos = cos[:seq_len, :][None, None, :, :]
    sin = sin[:seq_len, :][None, None, :, :]

    # 3. Apply the rotary transformation.
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.astype(x.dtype)
