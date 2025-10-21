"""The Gemma3 model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import TransformerBlock
from normalization import RMSNorm
from rope import compute_rope_params


class Gemma3Model(nn.Module):
    """
    The main Gemma3 model class.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        assert (
            cfg["layer_types"] is not None
            and len(cfg["layer_types"]) == cfg["n_layers"]
        ), "Layer types must be specified for each layer."

        # Token embedding layer
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]]
        )

        # Final normalization and output head
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        self.cfg = cfg

        # Precompute RoPE parameters
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the global and local attention masks.
        """
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # Global mask: prevents attending to future tokens
        mask_global = torch.triu(ones, diagonal=1)

        # Mask for tokens that are too far in the past
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T

        # Local mask: combines the future mask and the far-past mask
        mask_local = mask_global | far_past

        return mask_global, mask_local

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len = input_ids.shape

        # Get token embeddings
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        # Create attention masks
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        # Pass through transformer blocks
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

        # Final normalization and output
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Generates a sequence of tokens.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the model's context length
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]

            # Get model predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample the next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_weights_into_gemma(model, param_config, params):
    """function to load weights to a Gemma3Model"""

    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    # Embedding weights
    if "model.embed_tokens.weight" in params:
        model.tok_emb.weight = assign(
            model.tok_emb.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )

    # Iterate over transformer layers
    for layer in range(param_config["n_layers"]):
        block = model.blocks[layer]
        att = block.att
        # Attention projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{layer}.self_attn.q_proj.weight"],
            f"model.layers.{layer}.self_attn.q_proj.weight",
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{layer}.self_attn.k_proj.weight"],
            f"model.layers.{layer}.self_attn.k_proj.weight",
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{layer}.self_attn.v_proj.weight"],
            f"model.layers.{layer}.self_attn.v_proj.weight",
        )
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{layer}.self_attn.o_proj.weight"],
            f"model.layers.{layer}.self_attn.o_proj.weight",
        )
        # QK normalization weights
        att.q_norm.scale = assign(
            att.q_norm.scale,
            params[f"model.layers.{layer}.self_attn.q_norm.weight"],
            f"model.layers.{layer}.self_attn.q_norm.weight",
        )
        att.k_norm.scale = assign(
            att.k_norm.scale,
            params[f"model.layers.{layer}.self_attn.k_norm.weight"],
            f"model.layers.{layer}.self_attn.k_norm.weight",
        )
        # Feed forward weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{layer}.mlp.gate_proj.weight"],
            f"model.layers.{layer}.mlp.gate_proj.weight",
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{layer}.mlp.up_proj.weight"],
            f"model.layers.{layer}.mlp.up_proj.weight",
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{layer}.mlp.down_proj.weight"],
            f"model.layers.{layer}.mlp.down_proj.weight",
        )
        # LayerNorm weights
        block.input_layernorm.scale = assign(
            block.input_layernorm.scale,
            params[f"model.layers.{layer}.input_layernorm.weight"],
            f"model.layers.{layer}.input_layernorm.weight",
        )
        block.post_attention_layernorm.scale = assign(
            block.post_attention_layernorm.scale,
            params[f"model.layers.{layer}.post_attention_layernorm.weight"],
            f"model.layers.{layer}.post_attention_layernorm.weight",
        )
        # Pre‑ and post‑feed forward norms
        pre_key = f"model.layers.{layer}.pre_feedforward_layernorm.weight"
        post_key = f"model.layers.{layer}.post_feedforward_layernorm.weight"
        if pre_key in params:
            block.pre_feedforward_layernorm.scale = assign(
                block.pre_feedforward_layernorm.scale,
                params[pre_key],
                pre_key,
            )
        if post_key in params:
            block.post_feedforward_layernorm.scale = assign(
                block.post_feedforward_layernorm.scale,
                params[post_key],
                post_key,
            )

    # Final LayerNorm
    if "model.norm.weight" in params:
        model.final_norm.scale = assign(
            model.final_norm.scale,
            params["model.norm.weight"],
            "model.norm.weight",
        )
    # Output head
    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight,
            params["lm_head.weight"],
            "lm_head.weight",
        )
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
