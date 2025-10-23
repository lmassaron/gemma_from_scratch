"""RMSNorm normalization layer implementation."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.

    RMSNorm is a simplification of LayerNorm. It normalizes the activations of a layer
    by their root mean square, and then scales the result with a learnable parameter.
    Unlike LayerNorm, it does not perform mean-centering, which makes it
    computationally more efficient. This efficiency gain, ranging from
    7% to 64%, has led to its adoption in many modern large language models,
    such as the LLaMA and Gemma families.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        """
        Initializes the RMSNorm layer.

        Args:
            emb_dim (int): The embedding dimension, which is the size of the last dimension
                         of the input tensor. Normalization is applied across this dimension.
            eps (float, optional): A small value added to the denominator for
                                 numerical stability to prevent division by zero.
                                 Defaults to 1e-6.
            bias (bool, optional): If True, a learnable additive bias (shift) is included.
                                 Defaults to False.
        """
        # Call the constructor of the parent class (nn.Module)
        super().__init__()

        # Store the epsilon value for the forward pass.
        self.eps = eps

        # 'scale' is the learnable multiplicative parameter, often referred to as 'gamma' or 'g'
        # in normalization literature. It allows the network to learn the optimal
        # scale for the normalized outputs.
        # It is initialized to a tensor of zeros with the size of the embedding dimension.
        # The (1 + self.scale) formulation is used in the forward pass to initialize the
        # scaling factor to 1, which can help with training stability at the beginning.
        # This specific implementation detail matches the original Gemma3 implementation.
        self.scale = nn.Parameter(torch.zeros(emb_dim))

        # 'shift' is the learnable additive parameter, often referred to as 'beta'.
        # It is only created if the 'bias' argument is set to True.
        # This allows the network to learn an optimal offset for the normalized outputs.
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): The input tensor to be normalized. The normalization is
                              applied over the last dimension.

        Returns:
            torch.Tensor: The normalized tensor, with the same shape and data type as the input.
        """
        # Store the original data type of the input tensor.
        input_dtype = x.dtype

        # The core normalization computation is performed in float32 for improved
        # numerical stability. Operations like squaring and taking the mean can lead
        # to precision loss or overflow/underflow issues with lower precision types
        # like float16 or bfloat16.
        x_f = x.float()

        # 1. Compute the Root Mean Square (RMS) of the input tensor.
        #    - x_f.pow(2): Square each element of the tensor.
        #    - .mean(dim=-1, keepdim=True): Compute the mean of the squared values
        #      along the last dimension. 'keepdim=True' ensures the output tensor
        #      retains this dimension with a size of 1, which is necessary for broadcasting.
        var = x_f.pow(2).mean(dim=-1, keepdim=True)

        # 2. Normalize the input tensor.
        #    - torch.rsqrt(var + self.eps): Calculate the reciprocal of the square root
        #      of the variance plus epsilon. This is equivalent to 1 / sqrt(var + eps).
        #    - x_f * ...: Multiply the input tensor by this reciprocal square root.
        #      This scales the input tensor, effectively normalizing it.
        x_norm = x_f * torch.rsqrt(var + self.eps)

        # 3. Apply the learnable scale and optional shift.
        #    - (1.0 + self.scale.float()): The learnable scaling factor. Since 'scale'
        #      was initialized to zeros, this starts as 1.0.
        #    - The multiplication is broadcast across the tensor.
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            # If a learnable shift is configured, add it to the scaled output.
            out = out + self.shift.float()

        # Convert the output tensor back to the original input data type.
        return out.to(input_dtype)
