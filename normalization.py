"""RMSNorm normalization layer implementation."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        """
        Initializes the RMSNorm layer.

        Args:
            emb_dim (int): The embedding dimension.
            eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
            bias (bool, optional): Whether to include a bias term. Defaults to False.
        """
        super().__init__()
        self.eps = eps
        # The scale parameter is initialized to zeros, and we use (1 + self.scale)
        # during the forward pass to match the original Gemma3 implementation.
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # The computation is done in float32 for stability, as in the original implementation.
        input_dtype = x.dtype
        x_f = x.float()

        # Compute the variance and normalize
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)

        # Apply the scale and shift
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)
