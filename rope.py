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

    RoPE is a method for encoding positional information in a way that is sensitive
    to relative positions. Instead of adding positional information, it rotates existing
    embeddings based on their position. This is done by treating pairs of dimensions
    as a complex number and rotating them in a 2D plane.

    This function pre-computes the cosine and sine values that will be used to
    perform these rotations for all possible positions up to `context_length`.

    Args:
        head_dim (int): The dimension of each attention head's query or key vectors.
        theta_base (int, optional): The base for the theta value, which controls the
                                  periodicity of the rotations. Defaults to 10_000.
        context_length (int, optional): The maximum sequence length for which to
                                      pre-compute the embeddings. Defaults to 4096.
        dtype (torch.dtype, optional): The data type for the computed parameters.
                                     Defaults to torch.float32.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the pre-computed
                                         cosine and sine parameters.
    """
    # RoPE works by pairing up dimensions, so the head dimension must be even.
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 1. Compute the inverse frequencies for the rotations.
    #    Each pair of dimensions will rotate at a different frequency.
    #    - `torch.arange(0, head_dim, 2, ...)`: Creates a sequence [0, 2, 4, ..., head_dim-2].
    #    - `... / head_dim`: Normalizes these values to be in the range [0, 1).
    #    - `theta_base ** (...)`: Raises the base to these powers, creating a geometric
    #      progression of frequencies. Lower dimensions (earlier in the sequence)
    #      get higher frequencies (rotate faster), while higher dimensions get
    #      lower frequencies (rotate slower).
    #    - `1.0 / ...`: Takes the inverse to get the final frequencies.
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # 2. Generate position indices from 0 to context_length - 1.
    positions = torch.arange(context_length, dtype=dtype)

    # 3. Compute the angles for the rotary embeddings.
    #    This is done by taking the outer product of positions and inverse frequencies.
    #    The result is a matrix of shape (context_length, head_dim / 2), where each
    #    element `angles[m, i]` is `m * inv_freq[i]`. This is the rotation angle
    #    for position `m` and dimension pair `i`.
    angles = positions[:, None] * inv_freq[None, :]

    # 4. Duplicate the angles for each pair of dimensions.
    #    The same rotation angle is applied to both elements in a pair.
    #    This changes the shape to (context_length, head_dim).
    angles = torch.cat([angles, angles], dim=1)

    # 5. Precompute the sine and cosine values for these angles.
    #    This is an optimization to avoid recomputing them in every forward pass.
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Return the pre-computed values. They will be used to apply the rotations.
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.

    This function takes an input tensor (e.g., queries or keys from a self-attention
    mechanism) and applies the pre-computed rotations based on position.

    The rotation is applied to pairs of features. For a vector [x1, x2, x3, x4, ...],
    it rotates (x1, x2) by an angle theta_1, (x3, x4) by an angle theta_2, and so on.
    The rotation formula for a 2D vector (a, b) is:
    a' = a * cos(theta) - b * sin(theta)
    b' = b * cos(theta) + a * sin(theta)
    This function implements this logic in a vectorized way.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        cos (torch.Tensor): The precomputed cosine values from `compute_rope_params`.
        sin (torch.Tensor): The precomputed sine values from `compute_rope_params`.

    Returns:
        torch.Tensor: The tensor with RoPE applied, having the same shape as the input.
    """
    # Get the dimensions from the input tensor.
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 1. Split the input tensor's last dimension into two halves.
    #    This separates the 'a' and 'b' components for the 2D rotations.
    x1 = x[..., : head_dim // 2]  # First half (e.g., x1, x3, x5, ...)
    x2 = x[..., head_dim // 2 :]  # Second half (e.g., x2, x4, x6, ...)

    # 2. Adjust the shapes of sin and cos for broadcasting.
    #    We select the embeddings for the actual sequence length of the input `x`.
    #    Then we add batch and head dimensions (as singletons) so they can be
    #    broadcasted to match the input tensor's shape.
    #    Final shape: (1, 1, seq_len, head_dim)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 3. Apply the rotary transformation in a vectorized way.
    #    This corresponds to the rotation formulas mentioned above.
    #    - `rotated = torch.cat((-x2, x1), dim=-1)`: This creates a new tensor where
    #      the first half is `-x2` and the second half is `x1`. This is a clever
    #      trick to prepare the terms for the rotation formulas. It aligns the `-b`
    #      term for the `a'` calculation and the `a` term for the `b'` calculation.
    #    - `x * cos`: Computes the `a * cos(theta)` and `b * cos(theta)` parts.
    #    - `rotated * sin`: Computes the `-b * sin(theta)` and `a * sin(theta)` parts.
    #    - The sum of these two gives the final rotated vectors `a'` and `b'`.
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # 4. Return the result, ensuring it has the same data type as the original input.
    return x_rotated.to(dtype=x.dtype)
