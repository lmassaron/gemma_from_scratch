# Implementing Gemma from Scratch in JAX/Flax: A Deep Dive

In the rapidly evolving landscape of Large Language Models (LLMs), understanding the internal mechanics is just as crucial as using the models themselves. While PyTorch remains the dominant framework for deep learning research, Google's **JAX** (along with the **Flax** neural network library) has emerged as a powerhouse for high-performance computing, offering incredible speed through XLA compilation and a rigorous functional programming paradigm.

This article explores the journey of porting a "Gemma from Scratch" implementation from PyTorch to JAX/Flax. We will dive deep into the specific architectural choices, the "JAX-onic" solutions for training loops, and how we bridged the gap between PyTorch weights and JAX inference.

---

## 1. The Core Architecture: Functional Modules with Flax

In Flax, models are defined as classes inheriting from `nn.Module`. Unlike PyTorch's object-oriented approach where layers contain their own state, Flax separates the **definition** of the compute graph from the **parameters** (state).

### The Model Structure
The `Gemma3Model` class (in `model.py`) defines the high-level architecture. We use the `setup` method to register sub-modules (like layers and embeddings) and `__call__` to define the forward pass.

One distinct feature of our implementation is the handling of **Rotary Positional Embeddings (RoPE)**. Instead of recomputing rotation frequencies on the fly, we precompute them using `jax.numpy`. JAX's powerful array manipulation allows us to handle these complex broadcasting operations efficiently.

```python
# From rope.py
def compute_rope_params(head_dim, theta_base, context_length, dtype):
    # ... computation of inverse frequencies ...
    angles = positions[:, None] * inv_freq[None, :]
    # Vectorized sine/cosine generation
    return jnp.cos(angles), jnp.sin(angles)
```

These precomputed `cos` and `sin` tables are passed down to the attention layers, ensuring purely functional data flow.

### Grouped Query Attention (GQA) via Einsum
The heart of Gemma is the attention mechanism. We implemented Grouped Query Attention (GQA), which is a middle ground between standard Multi-Head Attention and Multi-Query Attention.

In `layers.py`, we leverage `jax.numpy.einsum` to perform the attention calculations. `einsum` allows us to describe the tensor contractions using a concise notation, making the code mathematically readable and often more performant.

```python
# Attention scores calculation using Einstein summation
# b: batch, h: heads, q: query_len, k: key_len, d: head_dim
attn_scores = jnp.einsum('bhqd,bhkd->bhqk', queries, keys)
```

To handle the "grouped" aspect of GQA (where multiple query heads share a single key/value head), we used `jnp.repeat` to explicitly broadcast the key and value tensors to match the number of query heads before the dot product.

---

## 2. The Training Loop: Compilation and Scanning

The most significant shift when moving to JAX is the training loop. In PyTorch, you typically write a Python `for` loop that iterates over batches. In JAX, to fully utilize the XLA (Accelerated Linear Algebra) compiler, you want to compile the entire update step into a single optimized kernel using `@jax.jit`.

### Functional State Management
We use `flax.training.train_state.TrainState` to act as a container for the model's parameters, the optimizer state (managed by **Optax**), and the model's apply function. This immutable state object is passed into the training step and a *new* state is returned.

### Gradient Accumulation with `jax.lax.scan`
One of the most interesting challenges was implementing **Gradient Accumulation**. In PyTorch, you simply run the forward/backward pass multiple times without zeroing gradients. In JAX, because the function must be stateless and compiled, we cannot easily "accumulate" side effects.

The naive solution—looping in Python and calling a JIT-compiled function multiple times—is inefficient because it leaves the accumulation logic outside the XLA boundary.

Our solution was to use **`jax.lax.scan`**. This primitive allows us to loop *inside* the compiled function. We reshape our input batch from `(total_batch, ...)` to `(accumulation_steps, micro_batch, ...)` and scan over the first dimension.

```python
@jax.jit
def train_step(state, batch, accumulation_steps):
    # ...
    def accumulate_grads(accum_grads, micro_batch):
        # Compute gradients for one micro-batch
        loss, grads = grad_fn(state.params, micro_batch)
        # Sum gradients
        new_accum_grads = jax.tree.map(lambda x, y: x + y, accum_grads, grads)
        return new_accum_grads, loss

    # Scan over the micro-batches efficiently within XLA
    final_accum_grads, losses = jax.lax.scan(accumulate_grads, init_accum_grads, (inputs, targets))
    
    # Average and apply updates
    grads = jax.tree.map(lambda x: x / accumulation_steps, final_accum_grads)
    new_state = state.apply_gradients(grads=grads)
    return new_state, jnp.mean(losses)
```

This approach is extremely fast because the entire accumulation loop is fused into a single computation graph on the GPU.

---

## 3. Inference: Interoperability and Static Shapes

For the inference script (`inference_custom.py`), we had to tackle two main issues: loading weights trained in PyTorch and ensuring efficient text generation.

### Bridging PyTorch and JAX
Since the ecosystem is split, being able to load PyTorch `.pt` checkpoints into a JAX model is a superpower. The challenge is that PyTorch state dictionaries are flat (e.g., `model.layers.0.self_attn.q_proj.weight`), while Flax parameters are nested dictionaries (e.g., `params['blocks']['block_0']['att']['W_query']['kernel']`).

We wrote a robust `load_pytorch_weights` function that:
1.  **Traverses the JAX parameter tree.**
2.  **Maps keys:** It associates PyTorch layer names with our Flax module names (handling conventions like `block_0` vs `layers.0`).
3.  **Transposes Weights:** PyTorch Linear layers store weights as `(out_features, in_features)`, whereas Flax Dense layers expect `(in_features, out_features)`. We detect and transpose these kernels automatically.
4.  **Handles `torch.compile` artifacts:** We added logic to strip the `_orig_mod.` prefix that PyTorch 2.0+ adds to compiled models, ensuring our loader is future-proof.
5.  **Security:** We utilized `weights_only=True` in `torch.load` to mitigate pickle security risks, falling back only when necessary.

### JIT-Compiled Generation
JAX hates dynamic shapes. A text generation loop that grows the input sequence one token at a time (`1 -> 2 -> 3...`) forces JAX to recompile the graph at every step, which is disastrous for performance.

To solve this, we implemented a **static buffer strategy**:
1.  We define a `MAX_LEN` (e.g., 2048).
2.  We initialize a buffer of this fixed size.
3.  The `forward_pass` is JIT-compiled once for this fixed shape `(1, MAX_LEN)`.
4.  We use padding and masking (handled naturally by the causal mask) to ignore the "future" empty slots in the buffer.

This allows the generation loop to run at full speed without triggering recompilation.

---

## Conclusion

Reimplementing Gemma in JAX was not just a translation exercise; it was a lesson in thinking functionally. By embracing JAX's primitives like `lax.scan` and `einsum`, and adhering to strict static shape requirements, we built a model that is not only correct but highly optimized for modern accelerators.

The result is a codebase that allows you to train utilizing the full power of JAX's gradient transformations and infer using weights seamlessly imported from the PyTorch world.

*The full code for this project is available in the `jax_experiments/` directory.*