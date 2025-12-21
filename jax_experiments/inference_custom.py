"""
Inference script for JAX/Flax Gemma model.
"""

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress verbose TF/XLA logs
import time

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken
from flax.serialization import from_bytes
from flax.core import unfreeze, freeze

# Ensure we can import from the package
try:
    from jax_experiments.model import Gemma3Model
    from jax_experiments.config import GEMMA3_CONFIG_CUSTOM
except ImportError:
    # If running directly from within the folder, adjust path or use relative
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from jax_experiments.model import Gemma3Model
    from jax_experiments.config import GEMMA3_CONFIG_CUSTOM

enc = tiktoken.get_encoding("gpt2")


def load_pytorch_weights(path, jax_params):
    import torch

    print(f"Loading PyTorch weights from {path}...")
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(
            f"Warning: Failed to load with weights_only=True ({e}). Retrying with weights_only=False."
        )
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

    # Normalize keys: remove _orig_mod. prefix if present
    # This handles checkpoints from torch.compile()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # Unfreeze jax_params to mutable dict
    new_params = unfreeze(jax_params)

    # Helper to set and transpose if needed
    def set_param(jax_path, torch_names, transpose=False):
        # Allow multiple possible torch names
        if isinstance(torch_names, str):
            torch_names = [torch_names]

        found_name = None
        for name in torch_names:
            if name in state_dict:
                found_name = name
                break

        if not found_name:
            print(
                f"Warning: None of {torch_names} found in state_dict. Keeping random initialization for {jax_path[-1]}"
            )
            return

        w = state_dict[found_name].float().numpy()
        if transpose:
            w = w.T

        # Navigate and set
        curr = new_params
        for key in jax_path[:-1]:
            if key not in curr:
                # print(f"Path in JAX model doesn't exist: {key}")
                return
            curr = curr[key]

        last_key = jax_path[-1]
        if last_key not in curr:
            print(f"Leaf key {last_key} not found in JAX params")
            return

        # Check shape compatibility
        target_shape = curr[last_key].shape
        if w.shape != target_shape:
            print(
                f"Shape mismatch for {found_name}: torch {w.shape} vs jax {target_shape}"
            )
            return

        curr[last_key] = jnp.array(w)
        # print(f"Loaded {found_name} -> {'.'.join(jax_path)}")

    # Detect convention
    is_hf = "model.embed_tokens.weight" in state_dict
    is_scratch = "tok_emb.weight" in state_dict

    if is_hf:
        print("Detected HuggingFace/Gemma naming convention.")
    elif is_scratch:
        print("Detected custom Gemma-scratch naming convention.")
    else:
        print("Warning: Unknown naming convention. Attempting fuzzy match or both.")

    # Embeddings
    set_param(["tok_emb", "embedding"], ["model.embed_tokens.weight", "tok_emb.weight"])

    # Layers
    # We find how many blocks by checking keys in new_params
    block_keys = [k for k in new_params.keys() if k.startswith("block_")]
    n_layers = len(block_keys)

    for i in range(n_layers):
        jax_block = f"block_{i}"

        # Attention
        # HF: model.layers.{i}.self_attn.q_proj.weight
        # Scratch: blocks.{i}.att.W_query.weight
        set_param(
            [jax_block, "att", "W_query", "kernel"],
            [
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"blocks.{i}.att.W_query.weight",
            ],
            transpose=True,
        )
        set_param(
            [jax_block, "att", "W_key", "kernel"],
            [
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"blocks.{i}.att.W_key.weight",
            ],
            transpose=True,
        )
        set_param(
            [jax_block, "att", "W_value", "kernel"],
            [
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"blocks.{i}.att.W_value.weight",
            ],
            transpose=True,
        )
        set_param(
            [jax_block, "att", "out_proj", "kernel"],
            [
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"blocks.{i}.att.out_proj.weight",
            ],
            transpose=True,
        )

        # Norms inside Attention (QK Norm)
        set_param(
            [jax_block, "att", "q_norm_layer", "scale"],
            [
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"blocks.{i}.att.q_norm.scale",
            ],
        )
        set_param(
            [jax_block, "att", "k_norm_layer", "scale"],
            [
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"blocks.{i}.att.k_norm.scale",
            ],
        )

        # MLP
        # HF: gate_proj, up_proj, down_proj
        # Scratch: fc1 (gate), fc2 (up), fc3 (down)
        set_param(
            [jax_block, "ff", "fc1", "kernel"],
            [f"model.layers.{i}.mlp.gate_proj.weight", f"blocks.{i}.ff.fc1.weight"],
            transpose=True,
        )
        set_param(
            [jax_block, "ff", "fc2", "kernel"],
            [f"model.layers.{i}.mlp.up_proj.weight", f"blocks.{i}.ff.fc2.weight"],
            transpose=True,
        )
        set_param(
            [jax_block, "ff", "fc3", "kernel"],
            [f"model.layers.{i}.mlp.down_proj.weight", f"blocks.{i}.ff.fc3.weight"],
            transpose=True,
        )

        # LayerNorms
        # HF: weight, Scratch: scale
        set_param(
            [jax_block, "input_layernorm", "scale"],
            [
                f"model.layers.{i}.input_layernorm.weight",
                f"blocks.{i}.input_layernorm.scale",
            ],
        )
        set_param(
            [jax_block, "post_attention_layernorm", "scale"],
            [
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"blocks.{i}.post_attention_layernorm.scale",
            ],
        )
        set_param(
            [jax_block, "pre_feedforward_layernorm", "scale"],
            [
                f"model.layers.{i}.pre_feedforward_layernorm.weight",
                f"blocks.{i}.pre_feedforward_layernorm.scale",
            ],
        )
        set_param(
            [jax_block, "post_feedforward_layernorm", "scale"],
            [
                f"model.layers.{i}.post_feedforward_layernorm.weight",
                f"blocks.{i}.post_feedforward_layernorm.scale",
            ],
        )

    # Final Norm
    set_param(["final_norm", "scale"], ["model.norm.weight", "final_norm.scale"])

    # Output Head
    # HF: lm_head.weight
    # Scratch: out_head.weight
    # Both need transpose

    # First check if we have explicit weights
    has_head_weight = "lm_head.weight" in state_dict or "out_head.weight" in state_dict

    if has_head_weight:
        set_param(
            ["out_head", "kernel"],
            ["lm_head.weight", "out_head.weight"],
            transpose=True,
        )
    else:
        # Weight tying
        print(
            "Weight tying detected (output head weights missing). Sharing embeddings."
        )
        if "tok_emb" in new_params and "embedding" in new_params["tok_emb"]:
            # embedding is (vocab, dim). Dense kernel expects (dim, vocab).
            new_params["out_head"]["kernel"] = new_params["tok_emb"]["embedding"].T
        else:
            print("Error: Could not perform weight tying, embeddings not found/loaded.")

    return freeze(new_params)


def generate(
    sentence,
    model,
    params,
    tokenizer,
    max_new_tokens=200,
    temperature=1.0,
    top_k=None,
    seed=0,
):
    print(f"Generating for: '{sentence}'")
    # Tokenize
    input_ids = tokenizer.encode_ordinary(sentence)
    initial_len = len(input_ids)

    # We use a fixed buffer size to enable JIT compilation
    # Total capacity = initial + max_new
    # We round up to a nice power of 2 or just use a sufficiently large buffer
    MAX_LEN = 2048
    if initial_len + max_new_tokens > MAX_LEN:
        print(
            f"Warning: Sequence length ({initial_len + max_new_tokens}) exceeds buffer ({MAX_LEN}). Truncating generation."
        )
        max_new_tokens = MAX_LEN - initial_len

    # Prepare buffer
    # Initialize with zeros (padding)
    buffer = np.zeros((1, MAX_LEN), dtype=np.int32)
    buffer[0, :initial_len] = input_ids

    rng = jax.random.PRNGKey(seed)

    # JIT-compiled forward function for the fixed shape
    @jax.jit
    def forward_pass(params, ids):
        # ids shape: (1, MAX_LEN)
        return model.apply({"params": params}, ids)

    current_len = initial_len

    # Generation loop
    for _ in range(max_new_tokens):
        # We pass the full buffer. The model's causal mask prevents
        # attending to the padding (future) tokens, so the valid tokens
        # are processed correctly.

        # Note: We must ensure we don't pass garbage in the "future" slots
        # if the model happened to be bidirectional, but Gemma is causal.
        # Zeros are fine.

        input_jax = jnp.array(buffer)

        logits = forward_pass(params, input_jax)

        # Extract logits for the last VALID token
        # logits shape: (1, MAX_LEN, vocab)
        next_token_logits = logits[0, current_len - 1, :]

        # Temperature
        if temperature == 0.0:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        else:
            next_token_logits = next_token_logits / temperature

            # Top-k sampling
            if top_k is not None:
                # We do top-k on CPU (numpy) to avoid dynamic shapes/sort on GPU if desired,
                # but for single token it's fast enough on GPU usually.
                # However, to match previous logic and keep it simple:
                logits_np = np.array(next_token_logits)
                ind = np.argpartition(logits_np, -top_k)[-top_k:]
                mask = np.ones_like(logits_np, dtype=bool)
                mask[ind] = False
                logits_np[mask] = -np.inf
                next_token_logits = jnp.array(logits_np)

            # Sampling
            rng, key = jax.random.split(rng)
            next_token = jax.random.categorical(key, next_token_logits, axis=-1)

        # Update buffer
        token_id = int(next_token)
        if current_len < MAX_LEN:
            buffer[0, current_len] = token_id
            current_len += 1

        # Check EOS
        if tokenizer.eot_token is not None and token_id == tokenizer.eot_token:
            break

    # Decode only the valid part
    output_ids = buffer[0, :current_len].tolist()
    return tokenizer.decode(output_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Gemma model (JAX)."
    )
    parser.add_argument(
        "--model-path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the saved model params file. If None, initializes random weights.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls randomness. Lower is more deterministic.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sample from the top K most likely tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set a fixed random seed.",
    )

    args = parser.parse_args()

    # Device Info
    print(f"JAX Devices: {jax.devices()}")

    # Initialize Model
    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)
    rng = jax.random.PRNGKey(args.seed)

    # Init dummy params to get structure
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    params = model.init(rng, dummy_input)["params"]

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        if args.model_path.endswith(".pt") or args.model_path.endswith(".pth"):
            params = load_pytorch_weights(args.model_path, params)
        else:
            with open(args.model_path, "rb") as f:
                params = from_bytes(params, f.read())
    else:
        print(
            "No model path provided or file not found. Using random initialized weights."
        )

    test_sentences = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "A boy told his sister a bedtime story about a flying cat",
    ]

    for k, test_sentence in enumerate(test_sentences):
        print(f"\n{'-' * 64}")
        print(f"{k + 1:2d}. input sentence: {test_sentence}")
        start = time.time()
        generated = generate(
            test_sentence,
            model,
            params,
            enc,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        end = time.time()
        print(f"Generated output:\n{generated}")
        print(f"/nTime taken: {end - start:.2f}s")
        print(f"{'-' * 64}\n")
