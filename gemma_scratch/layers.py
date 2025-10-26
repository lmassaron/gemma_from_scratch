"""Core layers for the Gemma3 transformer model."""

import torch
from torch import nn

# Importing the custom RMSNorm and RoPE implementations
from gemma_scratch.normalization import RMSNorm
from gemma_scratch.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.

    GQA is an attention mechanism that strikes a balance between standard
    Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
    - In MHA, each query head has its own key (K) and value (V) head.
    - In MQA, all query heads share a single K and V head.
    - In GQA, multiple query heads are grouped together, and each group shares
      a single K and V head. This reduces the number of parameters and the
      computational load for K and V projections compared to MHA, while often
      maintaining better performance than MQA.
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
        """
        Initializes the Grouped Query Attention module.

        Args:
            d_in (int): The input dimension of the model (embedding dimension).
            num_heads (int): The total number of query heads.
            num_kv_groups (int): The number of groups for key/value heads. `num_heads` must be divisible by this.
            head_dim (int, optional): The dimension of each attention head. If None, it's inferred from `d_in` and `num_heads`.
            qk_norm (bool, optional): If True, applies RMSNorm to queries and keys before attention. Defaults to False.
            query_pre_attn_scalar (float, optional): A custom scaling factor for queries. If None, defaults to `head_dim**-0.5`.
            dtype (torch.dtype, optional): The data type for the layer's weights.
        """
        super().__init__()
        # Ensure that the number of query heads is a multiple of the number of K/V groups.
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # Calculate the number of query heads per K/V group.
        self.group_size = num_heads // num_kv_groups

        # Determine the dimension of each head if not explicitly provided.
        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        # The total output dimension from all heads combined.
        self.d_out = num_heads * head_dim

        # === Linear projections for query, key, and value ===
        # Query projection: Maps input to the combined dimension of all query heads.
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # Key projection: Maps input to the combined dimension of all key heads (one per group).
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        # Value projection: Maps input to the combined dimension of all value heads (one per group).
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        # Output projection: Maps the concatenated attention outputs back to the model's input dimension.
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # Optional RMSNorm for queries (Q) and keys (K). Normalizing them can improve training stability.
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # Scaling factor for the query before the attention score calculation.
        # This is a standard practice to prevent the dot products from growing too large,
        # which can lead to vanishing gradients in the softmax function.
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = head_dim**-0.5

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Grouped Query Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).
            mask (torch.Tensor): Attention mask to prevent attending to certain positions.
            cos (torch.Tensor): Pre-computed cosine values for RoPE.
            sin (torch.Tensor): Pre-computed sine values for RoPE.

        Returns:
            torch.Tensor: The output tensor after attention, of shape (batch_size, num_tokens, d_in).
        """
        # Get the batch size and sequence length from the input tensor.
        b, num_tokens, _ = x.shape

        # 1. Apply linear projections to get the queries, keys, and values.
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 2. Reshape Q, K, V tensors to separate the heads.
        #    The shape becomes (batch_size, num_heads, num_tokens, head_dim).
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # 3. Apply optional normalization to queries and keys.
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # 4. Apply Rotary Positional Embeddings (RoPE) to queries and keys.
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # 5. Repeat keys and values to match the number of query heads for GQA.
        #    Each K/V head is shared across `self.group_size` query heads.
        #    `repeat_interleave` duplicates the K/V heads along the head dimension.
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 6. Scale queries before computing attention scores.
        queries = queries * self.scaling

        # 7. Compute attention scores (dot product between queries and keys).
        #    - Q shape: (b, num_heads, num_tokens, head_dim)
        #    - K.T shape: (b, num_heads, head_dim, num_tokens)
        #    - Result shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # Apply the attention mask (e.g., to prevent attending to future tokens).
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        # Apply softmax to get attention weights.
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 8. Compute the context vector (weighted sum of values).
        #    - weights shape: (b, num_heads, num_tokens, num_tokens)
        #    - V shape: (b, num_heads, num_tokens, head_dim)
        #    - Result shape: (b, num_heads, num_tokens, head_dim)
        context = attn_weights @ values

        # 9. Reshape the context vector back to the original tensor format.
        #    - `transpose(1, 2)`: Swaps heads and tokens dimensions -> (b, num_tokens, num_heads, head_dim)
        #    - `reshape(...)`: Merges the head and head_dim dimensions -> (b, num_tokens, d_out)
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)

        # 10. Apply the final output projection.
        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    The feed-forward network (FFN) in a transformer block.

    This specific implementation uses a SwiGLU-like architecture (using GELU instead of Swish),
    which is common in modern transformers like Llama and Gemma. It involves three linear
    projections and a gated activation.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the FeedForward network.

        Args:
            cfg (dict): A configuration dictionary containing model parameters like
                        'emb_dim', 'hidden_dim', and 'dtype'.
        """
        super().__init__()
        # The first linear layer, often called the "up" projection, expands the dimension.
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        # The second linear layer also expands the dimension and acts as the "gate" in the SwiGLU-like structure.
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        # The third linear layer, the "down" projection, maps the dimension back to the original embedding size.
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward network.

        The computation is: `Output = W3(GELU(W1 * x) * (W2 * x))`
        """
        # Apply the first and second linear projections to the input.
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        # Apply the GELU activation function to the first projection.
        # The 'tanh' approximation is a faster variant of GELU.
        activated_x = nn.functional.gelu(x_fc1, approximate="tanh")
        # Perform element-wise multiplication (the "gating" mechanism).
        x = activated_x * x_fc2
        # Apply the final "down" projection.
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block, which contains an attention mechanism and a feed-forward network.
    This block uses a specific pre/post normalization scheme with residual connections.
    """

    def __init__(self, cfg: dict, attn_type: str):
        super().__init__()
        self.attn_type = attn_type

        # Initialize the Grouped Query Attention mechanism.
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )

        # Initialize the Feed-forward network.
        self.ff = FeedForward(cfg)

        # === Normalization Layers ===
        # This implementation uses a somewhat unique normalization style where normalization
        # is applied both *before* the main module (pre-norm) and *after* it (post-norm)
        # but before the residual connection is added.
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
        """
        Forward pass for the Transformer Block.
        """
        # --- Attention Sub-block ---
        # 1. Store the input for the residual connection (skip connection).
        shortcut = x
        # 2. Pre-normalize the input before the attention module.
        x = self.input_layernorm(x)

        # 3. Select the appropriate mask and RoPE parameters based on the attention type.
        #    This allows the model to switch between global and local (sliding window) attention.
        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:  # "global_attention"
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        # 4. Pass the normalized input through the attention module.
        x_attn = self.att(x, attn_mask, cos, sin)
        # 5. Post-normalize the output of the attention module.
        x_attn = self.post_attention_layernorm(x_attn)
        # 6. Add the residual connection.
        x = shortcut + x_attn

        # --- Feed-Forward Sub-block ---
        # 1. Store the output of the attention block for the next residual connection.
        shortcut = x
        # 2. Pre-normalize the input before the FFN module.
        x_ffn = self.pre_feedforward_layernorm(x)
        # 3. Pass the normalized input through the FFN module.
        x_ffn = self.ff(x_ffn)
        # 4. Post-normalize the output of the FFN module.
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        # 5. Add the second residual connection.
        x = shortcut + x_ffn

        return x
