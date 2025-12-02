"""RMSNorm normalization layer implementation in JAX/Flax."""

import jax
import jax.numpy as jnp
from flax import linen as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.
    """
    emb_dim: int
    eps: float = 1e-6
    bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 'scale' is the learnable multiplicative parameter.
        # Initialized to zeros as per original implementation (1 + scale).
        self.scale = self.param('scale', nn.initializers.zeros, (self.emb_dim,), jnp.float32)

        if self.bias:
            self.shift = self.param('shift', nn.initializers.zeros, (self.emb_dim,), jnp.float32)
        else:
            self.shift = None

    def __call__(self, x):
        """
        Forward pass for RMSNorm.
        """
        input_dtype = x.dtype
        x_f = x.astype(jnp.float32)

        # 1. Compute the Root Mean Square (RMS) of the input tensor.
        var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)

        # 2. Normalize the input tensor.
        x_norm = x_f * jax.lax.rsqrt(var + self.eps)

        # 3. Apply the learnable scale and optional shift.
        # The (1 + self.scale) formulation.
        out = x_norm * (1.0 + self.scale)
        
        if self.shift is not None:
            out = out + self.shift

        return out.astype(input_dtype)
