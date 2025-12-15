"""
Inference script for JAX/Flax Gemma model.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken
from flax.serialization import from_bytes

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

def generate(
    sentence, 
    model, 
    params, 
    tokenizer, 
    max_new_tokens=200, 
    temperature=1.0, 
    top_k=None,
    seed=0
):
    print(f"Generating for: '{sentence}'")
    # Tokenize
    input_ids = tokenizer.encode_ordinary(sentence)
    input_ids = jnp.array([input_ids], dtype=jnp.int32) # (1, seq_len) 
    
    rng = jax.random.PRNGKey(seed) 
    
    # Generation loop
    cur_ids = input_ids
    
    # We can JIT the single step for performance
    @jax.jit
    def next_token_step(params, ids, rng_key):
        # Crop context if needed within the model call or here?
        # Ideally we pass the last `context_length` tokens.
        ctx_len = GEMMA3_CONFIG_CUSTOM["context_length"]
        # JAX array slicing must be static or handled carefully in JIT.
        # Since shapes change, we might re-compile every step if we JIT the whole thing with dynamic shape.
        # For this simple inference, we might NOT JIT the outer loop logic, 
        # but JIT the model forward pass with a fixed size or just accept recompilation (slow).
        # Better: use a fixed window size for the model call if possible, or just eager execution for simplicity.
        # Eager execution for variable length inference is often acceptable for debugging/small scale.
        
        # Let's just JIT the model apply for the current shape. 
        # Since shape changes every step (+1 token), it will recompile every step.
        # To avoid this, we usually use padding and `jax.lax.scan` or a `kv_cache`.
        # For "from scratch" simplicity without kv-cache:
        # We will just run eager or accept the overhead.
        
        logits = model.apply({'params': params}, ids)
        next_token_logits = logits[:, -1, :]
        return next_token_logits

    for _ in range(max_new_tokens):
        # Crop context
        ctx_len = GEMMA3_CONFIG_CUSTOM["context_length"]
        if cur_ids.shape[1] > ctx_len:
            cond_ids = cur_ids[:, -ctx_len:]
        else:
            cond_ids = cur_ids

        # Forward pass
        # We use the model.apply directly without JIT to avoid recompilation spam on dynamic shapes
        # unless we pad. For this demo, raw apply is fine.
        logits = model.apply({'params': params}, cond_ids)
        next_token_logits = logits[:, -1, :]
        
        # Temperature
        if temperature == 0.0:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        else:
            next_token_logits = next_token_logits / temperature
            
            # Top-k (Simple NumPy-based implementation for CPU/MPS inference efficiency without complex JAX ops)
            if top_k is not None:
                # Convert to numpy for flexible indexing
                logits_np = np.array(next_token_logits)
                # For each batch item (here 1) 
                for i in range(logits_np.shape[0]):
                    row = logits_np[i]
                    # Find top k
                    ind = np.argpartition(row, -top_k)[-top_k:]
                    # Set others to -inf
                    mask = np.ones_like(row, dtype=bool)
                    mask[ind] = False
                    row[mask] = -np.inf
                    logits_np[i] = row
                next_token_logits = jnp.array(logits_np)

            # Sampling
            rng, key = jax.random.split(rng)
            next_token = jax.random.categorical(key, next_token_logits, axis=-1)
        
        next_token = next_token[:, None]
        cur_ids = jnp.concatenate([cur_ids, next_token], axis=1)
        
        # Check EOS
        if tokenizer.eot_token is not None and next_token[0,0] == tokenizer.eot_token:
            break
            
    # Decode
    output_ids = cur_ids[0].tolist()
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
        with open(args.model_path, "rb") as f:
            params = from_bytes(params, f.read())
    else:
        print("No model path provided or file not found. Using random initialized weights.")

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
            seed=args.seed
        )
        end = time.time()
        print(f"Generated output:\n{generated}")
        print(f"Time taken: {end - start:.2f}s")
        print(f"{'-' * 64}\n")
