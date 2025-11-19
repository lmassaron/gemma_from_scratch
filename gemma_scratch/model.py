"""The Gemma3 model architecture."""

import torch
from torch import nn
import torch.nn.functional as F

# Importing the custom building blocks of the transformer.
from gemma_scratch.layers import TransformerBlock
from gemma_scratch.normalization import RMSNorm
from gemma_scratch.rope import compute_rope_params


class Gemma3Model(nn.Module):
    """
    The main Gemma3 model class.

    This class assembles the various components (embedding, transformer blocks,
    normalization, output head) into a full, functional decoder-only transformer model.
    It handles the forward pass for training, the generation process for inference,
    and the creation of attention masks.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the Gemma3 model.

        Args:
            cfg (dict): A configuration dictionary containing all model hyperparameters,
                        such as vocab size, number of layers, embedding dimension, etc.
        """
        super().__init__()
        # Ensure that the configuration specifies an attention type for each layer.
        # This allows for hybrid models with both global and sliding-window attention.
        assert (
            cfg["layer_types"] is not None
            and len(cfg["layer_types"]) == cfg["n_layers"]
        ), "Layer types must be specified for each layer."

        # Token embedding layer: maps input token IDs to dense vectors.
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        # A list of TransformerBlocks. Using nn.ModuleList ensures that all blocks
        # are properly registered as submodules of the model.
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]]
        )

        # Final normalization layer applied before the output head.
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        # Output head (or language model head): a linear layer that projects the final
        # transformer output back to the vocabulary space to get logits.
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        self.cfg = cfg

        # === Precompute Rotary Positional Embedding (RoPE) parameters ===
        # RoPE parameters are fixed and depend only on hyperparameters, so they
        # can be computed once during initialization instead of every forward pass.
        # Two sets of RoPE parameters are created: one for local (sliding window)
        # attention and one for global attention, each with a potentially different base.
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
        # register_buffer stores these tensors as part of the model's state,
        # ensuring they are moved to the correct device (e.g., GPU) with the model.
        # `persistent=False` means these buffers will not be saved in the model's
        # state_dict, as they can be re-computed from the config.
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the global and local attention masks.

        Masks are crucial for decoder-only transformers to control what information
        each token can "see". A `True` value in the mask indicates a position
        that should be masked out (i.e., not attended to).
        """
        # Create a square boolean matrix of all ones.
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # Global mask (causal mask): This is a standard upper-triangular mask
        # that prevents any token from attending to future tokens.
        # `torch.triu(..., diagonal=1)` creates a matrix with `True` values
        # in the upper triangle, starting from the first super-diagonal.
        mask_global = torch.triu(ones, diagonal=1)

        # Mask for sliding window attention: This mask should prevent attending to
        # two types of tokens: future tokens AND tokens that are too far in the past.
        # 1. Create a mask for tokens that are too far in the past.
        #    `torch.triu(..., diagonal=W)` creates a mask for positions more than
        #    W steps ahead. `.T` (transpose) flips this to mask positions more than
        #    W steps in the past.
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T

        # 2. The final local mask is the combination (logical OR) of the future mask
        #    and the far-past mask.
        mask_local = mask_global | far_past

        return mask_global, mask_local

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for training and evaluation.

        Args:
            input_ids (torch.Tensor): The input token IDs. Shape: (batch_size, seq_len).
            targets (torch.Tensor, optional): The target token IDs for loss calculation.
                                              Shape: (batch_size, seq_len).

        Returns:
            A tuple containing:
            - logits (torch.Tensor): The raw, unnormalized scores for each token in the vocabulary.
            - loss (torch.Tensor): The cross-entropy loss, or None if targets are not provided.
        """
        b, seq_len = input_ids.shape

        # 1. Get token embeddings and scale them. This scaling is a standard practice
        #    in many transformer models.
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        # 2. Create the attention masks for the current sequence length.
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        # 3. Pass the embeddings through the stack of transformer blocks.
        #    Each block applies its specific attention (global or local) using the
        #    corresponding mask and RoPE parameters.
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

        # 4. Apply final normalization and project to the vocabulary to get logits.
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        # 5. Compute the loss if target labels are provided.
        loss = None
        if targets is not None:
            # The cross_entropy function expects logits of shape (N, C) and targets of shape (N).
            # We reshape our logits from (batch, seq_len, vocab_size) to (batch*seq_len, vocab_size)
            # and targets from (batch, seq_len) to (batch*seq_len).
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
        eos_id: int = None,
    ) -> torch.Tensor:
        """
        Generates a sequence of new tokens autoregressively.

        Args:
            idx (torch.Tensor): The starting sequence of token IDs.
            max_new_tokens (int): The maximum number of tokens to generate.
            temperature (float): A scaling factor for the logits. Lower values make
                                 the output more deterministic, higher values make it more random.
            top_k (int, optional): If set, only the top `k` most likely tokens are considered
                                   for sampling at each step.

        Returns:
            torch.Tensor: The input sequence with the newly generated tokens appended.
        """
        # The generation loop runs for a fixed number of steps.
        for _ in range(max_new_tokens):
            # 1. Crop the context if it exceeds the model's supported context length.
            #    We only need to pass a sequence of at most `context_length` to the model.
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]

            # 2. Get the model's predictions (logits) for the next token.
            logits, _ = self(idx_cond)

            # 3. Focus only on the logits for the very last token in the sequence.
            # Previously, logits was 3D [Batch, Seq, Vocab]
            # Slicing ensures probs is [Batch, Vocab]
            logits = logits[:, -1, :]

            if temperature==0.0: # Greedy decoding
                # Just pick the single most likely token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else: # Probability sampling
                # Apply temperature scaling
                logits = logits / temperature

                # Optional top-k sampling: this technique truncates the probability distribution.
                if top_k is not None:
                    # Get the k-th largest logit value.
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # Set all logits smaller than this threshold to negative infinity,
                    # so they will have a probability of 0 after softmax.
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Convert logits to probabilities and sample the next token.
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # 3. Append the newly sampled token to the running sequence.
            idx = torch.cat((idx, idx_next), dim=1)

            # 4. Check for EOS (Stop generation)
            if eos_id is not None and idx_next.item() == eos_id:
                print("[EOS]")
                break

        return idx


def load_weights_into_gemma(model, param_config, params):
    """
    A utility function to load pre-trained weights from a dictionary-like object
    (e.g., loaded from a safetensors file) into a Gemma3Model instance.

    It maps the parameter names from the pre-trained checkpoint to the corresponding
    layers and attributes in our model definition.
    """

    def assign(left, right, tensor_name="unknown"):
        """A helper function to safely copy tensor data, with shape checking."""
        if left.shape != right.shape:
            # Raise an error if the model's weight shape and the loaded weight shape don't match.
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )

        # Use no_grad to prevent tracking this operation in the computation graph.
        with torch.no_grad():
            # `.copy_()` is an in-place operation that copies the data from `right` to `left`.
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:  # Convert numpy arrays or lists to tensors before copying.
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    # Embedding weights
    if "model.embed_tokens.weight" in params:
        model.tok_emb.weight = assign(
            model.tok_emb.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )

    # Iterate over each transformer layer and assign its weights.
    for layer in range(param_config["n_layers"]):
        block = model.blocks[layer]
        att = block.att
        # Attention projections (Query, Key, Value, Output)
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
        # QK normalization weights (the learnable 'scale' parameter in RMSNorm)
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
        # Feed forward weights (SwiGLU-style layers)
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
        # Pre- and post-feed forward norms (these might be optional)
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

    # Output head (language model head)
    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight,
            params["lm_head.weight"],
            "lm_head.weight",
        )
    else:
        # If the output head weights are not in the checkpoint, it implies weight tying.
        # Weight tying is a technique where the token embedding weights and the final
        # output head weights are shared. This saves a significant number of parameters.
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
