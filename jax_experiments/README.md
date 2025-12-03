I have created a jax directory containing a complete JAX/Flax implementation of the Gemma model project.
  This includes:

   * `jax/config.py`: Mirrored configuration from the original project.
   * `jax/model.py`: The Gemma3Model implemented using flax.linen.
   * `jax/layers.py`: Implementations of GroupedQueryAttention, FeedForward, and TransformerBlock in Flax.
   * `jax/normalization.py`: RMSNorm implementation in Flax.
   * `jax/rope.py`: Rotary Positional Embeddings using JAX operations.
   * `jax/tokenizer.py`: Identical tokenizer utility for compatibility.
   * `jax/train.py`: A full training loop using optax for optimization and jax.jit/jax.lax for
     high-performance training. It reuses the PyTorch DataLoader for efficient data loading but converts
     batches to JAX arrays for the training step.
