"""The Gemma3 model architecture in JAX/Flax."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import TransformerBlock
from .normalization import RMSNorm
from .rope import compute_rope_params


class Gemma3Model(nn.Module):
    """
    The main Gemma3 model class in JAX/Flax.
    """

    cfg: dict

    def setup(self):
        assert (
            self.cfg["layer_types"] is not None
            and len(self.cfg["layer_types"]) == self.cfg["n_layers"]
        ), "Layer types must be specified for each layer."

        self.tok_emb = nn.Embed(
            num_embeddings=self.cfg["vocab_size"],
            features=self.cfg["emb_dim"],
            dtype=self.cfg["dtype"],
        )

        self.blocks = [
            TransformerBlock(self.cfg, attn_type, name=f"block_{i}")
            for i, attn_type in enumerate(self.cfg["layer_types"])
        ]

        self.final_norm = RMSNorm(
            self.cfg["emb_dim"], eps=1e-6, dtype=self.cfg["dtype"]
        )
        self.out_head = nn.Dense(
            self.cfg["vocab_size"], use_bias=False, dtype=self.cfg["dtype"]
        )

        # Precompute RoPE
        # In JAX/Flax, we typically compute constants outside setup if they don't have params,
        # or we can compute them here and store them. Since they are constants, we can just compute them.
        # However, we can't store them as attributes easily if they are large arrays without making them variables.
        # But for RoPE params which are smallish (context_len * head_dim), it's fine.
        # Ideally, we should use `self.variable` with `cache` collection for immutable constants,
        # but recalculating them or passing them is also fine.
        # To match the PyTorch style where they are part of the model state (buffers),
        # we can put them in a variable collection.

        self.cos_local, self.sin_local = compute_rope_params(
            head_dim=self.cfg["head_dim"],
            theta_base=self.cfg["rope_local_base"],
            context_length=self.cfg["context_length"],
            dtype=jnp.float32,
        )
        self.cos_global, self.sin_global = compute_rope_params(
            head_dim=self.cfg["head_dim"],
            theta_base=self.cfg["rope_base"],
            context_length=self.cfg["context_length"],
            dtype=jnp.float32,
        )

    def _create_masks(self, seq_len):
        ones = jnp.ones((seq_len, seq_len), dtype=bool)

        # Global mask (causal)
        # jnp.triu gives upper triangle.
        # PyTorch: triu(ones, diagonal=1) -> upper triangle excluding diagonal.
        # Mask means "masked out" (True).
        # In PyTorch code: mask_global = torch.triu(ones, diagonal=1)
        # This puts True in upper triangle (future).
        mask_global = jnp.triu(ones, k=1)

        # Local mask
        # far_past = torch.triu(ones, diagonal=window).T
        # jnp.triu(..., k=window).T
        far_past = jnp.triu(ones, k=self.cfg["sliding_window"]).T

        mask_local = mask_global | far_past

        return mask_global, mask_local

    def __call__(self, input_ids, training=False):
        # input_ids: (batch, seq_len)
        b, seq_len = input_ids.shape

        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        mask_global, mask_local = self._create_masks(seq_len)

        # Expand masks for broadcasting over batch and heads
        # (1, 1, seq_len, seq_len)
        mask_global = mask_global[None, None, :, :]
        mask_local = mask_local[None, None, :, :]

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        # Cast to dtype before projection if needed, but Dense usually handles it.
        # The original code did x.to(dtype)
        logits = self.out_head(x.astype(self.cfg["dtype"]))

        return logits
