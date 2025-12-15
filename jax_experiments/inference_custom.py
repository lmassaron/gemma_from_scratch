"""
Inference script for JAX/Flax Gemma model.
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress verbose TF/XLA logs
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
    initial_len = len(input_ids)
    
    # We use a fixed buffer size to enable JIT compilation
    # Total capacity = initial + max_new
    # We round up to a nice power of 2 or just use a sufficiently large buffer
    MAX_LEN = 2048 
    if initial_len + max_new_tokens > MAX_LEN:
        print(f"Warning: Sequence length ({initial_len + max_new_tokens}) exceeds buffer ({MAX_LEN}). Truncating generation.")
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
        return model.apply({'params': params}, ids)

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
