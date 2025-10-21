"""Core layers for the Gemma3 transformer model."""

import torch
import torch.nn as nn

from normalization import RMSNorm
from rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.
    """

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: float = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Linear projections for query, key, and value
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # Optional normalization for queries and keys
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # Scaling factor for the query
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = head_dim**-0.5

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        # Apply linear projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply Rotary Positional Embeddings
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Repeat keys and values to match the number of query heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Scale queries
        queries = queries * self.scaling

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute the context vector
        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )

        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    The feed-forward network in a transformer block.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block, containing attention and feed-forward layers.
    """

    def __init__(self, cfg: dict, attn_type: str):
        super().__init__()
        self.attn_type = attn_type

        # Attention mechanism
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )

        # Feed-forward network
        self.ff = FeedForward(cfg)

        # Normalization layers
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask_global: torch.Tensor,
        mask_local: torch.Tensor,
        cos_global: torch.Tensor,
        sin_global: torch.Tensor,
        cos_local: torch.Tensor,
        sin_local: torch.Tensor,
    ) -> torch.Tensor:
        # Attention block with pre-normalization and residual connection
        shortcut = x
        x = self.input_layernorm(x)

        # Select the appropriate mask and RoPE parameters based on the attention type
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

        # Feed-forward block with pre-normalization and residual connection
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn

        return x
