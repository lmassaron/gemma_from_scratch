"""Core layers for the Gemma3 transformer model in JAX/Flax."""

import jax.numpy as jnp
from flax import linen as nn
from jax.nn import gelu

from .normalization import RMSNorm
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.
    """
    d_in: int
    num_heads: int
    num_kv_groups: int
    head_dim: int = None
    qk_norm: bool = False
    query_pre_attn_scalar: float = None
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        assert self.num_heads % self.num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )
        self.group_size = self.num_heads // self.num_kv_groups

        if self.head_dim is None:
            assert self.d_in % self.num_heads == 0
            self.head_dim_val = self.d_in // self.num_heads
        else:
            self.head_dim_val = self.head_dim
            
        self.d_out = self.num_heads * self.head_dim_val

        self.W_query = nn.Dense(self.d_out, use_bias=False, dtype=self.dtype)
        self.W_key = nn.Dense(self.num_kv_groups * self.head_dim_val, use_bias=False, dtype=self.dtype)
        self.W_value = nn.Dense(self.num_kv_groups * self.head_dim_val, use_bias=False, dtype=self.dtype)
        self.out_proj = nn.Dense(self.d_in, use_bias=False, dtype=self.dtype)

        if self.qk_norm:
            self.q_norm_layer = RMSNorm(self.head_dim_val, eps=1e-6, dtype=self.dtype)
            self.k_norm_layer = RMSNorm(self.head_dim_val, eps=1e-6, dtype=self.dtype)
        else:
            self.q_norm_layer = self.k_norm_layer = None

        if self.query_pre_attn_scalar is not None:
            self.scaling = self.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim_val**-0.5

    def __call__(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim_val).transpose(0, 2, 1, 3)
        keys = keys.reshape(b, num_tokens, self.num_kv_groups, self.head_dim_val).transpose(0, 2, 1, 3)
        values = values.reshape(b, num_tokens, self.num_kv_groups, self.head_dim_val).transpose(0, 2, 1, 3)

        if self.qk_norm:
            queries = self.q_norm_layer(queries)
            keys = self.k_norm_layer(keys)

        # Apply RoPE needs (batch, heads, seq, dim)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Repeat keys/values for GQA
        # keys shape: (b, kv_groups, seq, dim) -> (b, heads, seq, dim)
        keys = jnp.repeat(keys, self.group_size, axis=1)
        values = jnp.repeat(values, self.group_size, axis=1)

        queries = queries * self.scaling

        # Attention scores: Q @ K.T
        # Q: (b, heads, seq, dim)
        # K: (b, heads, seq, dim)
        # Result: (b, heads, seq, seq)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', queries, keys)
        
        if mask is not None:
             attn_scores = jnp.where(mask, -jnp.inf, attn_scores)

        attn_weights = nn.softmax(attn_scores, axis=-1)

        # Context: weights @ V
        context = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, values)

        # Reshape back: (b, seq, heads, dim) -> (b, seq, d_out)
        context = context.transpose(0, 2, 1, 3).reshape(b, num_tokens, self.d_out)

        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    The feed-forward network (FFN) in a transformer block.
    SwiGLU-like architecture.
    """
    cfg: dict

    def setup(self):
        self.fc1 = nn.Dense(self.cfg["hidden_dim"], use_bias=False, dtype=self.cfg["dtype"])
        self.fc2 = nn.Dense(self.cfg["hidden_dim"], use_bias=False, dtype=self.cfg["dtype"])
        self.fc3 = nn.Dense(self.cfg["emb_dim"], use_bias=False, dtype=self.cfg["dtype"])

    def __call__(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        activated_x = gelu(x_fc1, approximate=True) # JAX gelu approximate is tanh based
        x = activated_x * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block.
    """
    cfg: dict
    attn_type: str

    def setup(self):
        self.att = GroupedQueryAttention(
            d_in=self.cfg["emb_dim"],
            num_heads=self.cfg["n_heads"],
            num_kv_groups=self.cfg["n_kv_groups"],
            head_dim=self.cfg["head_dim"],
            qk_norm=self.cfg["qk_norm"],
            query_pre_attn_scalar=self.cfg["query_pre_attn_scalar"],
            dtype=self.cfg["dtype"],
        )
        self.ff = FeedForward(self.cfg)

        self.input_layernorm = RMSNorm(self.cfg["emb_dim"], eps=1e-6, dtype=self.cfg["dtype"])
        self.post_attention_layernorm = RMSNorm(self.cfg["emb_dim"], eps=1e-6, dtype=self.cfg["dtype"])
        self.pre_feedforward_layernorm = RMSNorm(self.cfg["emb_dim"], eps=1e-6, dtype=self.cfg["dtype"])
        self.post_feedforward_layernorm = RMSNorm(self.cfg["emb_dim"], eps=1e-6, dtype=self.cfg["dtype"])

    def __call__(
        self,
        x,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn

        return x
