# Presentation Guide: Notebooks 00–04 (Setup → Attention → Masking → GQA → RoPE)

---

## 🕐 Total Time for This Block: ~45 minutes
## 🧱 Building the "Vision" of the Model

> **Presenter Note:** This block establishes the *core vision machinery* of Gemma 3. Each notebook adds one new ingredient to how the model "sees" and "pays attention" to its input. Start slow, reinforce concepts visually.

---

## ──────────────────────────────────────────────
## Notebook 00: Setup & Tokenization (5 min)
### ──────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Vocab + embedding intro | 2 min |
| Embedding scaling | 2 min |
| Run notebook | 1 min |

### Teaching Points
- **What is tokenization?** Break text into subword units. Gemma 3 uses a BPE vocabulary of 262,144 tokens (including Unicode). Each token maps to a dense vector.
- **`sqrt(hidden_size)` scaling:** Embeddings are initialized small (~N(0, 0.02²)), but immediately multiplied by `sqrt(768) ≈ 27.7`. Why? To prevent the very first attention matrix from being tiny (vanishing gradients). This trick dates from [Dong et al. 2020](https://arxiv.org/abs/2002.08262), predating Gemma.
- **No mean subtraction in embeddings:** Unlike LayerNorm, embeddings are raw vectors — the first normalization happens at RMSNorm.

### 🚩 Common Misconception
> *"Tokenization is just splitting text into words."* — **Correct it:** Tokenizers use BPE (Byte-Pair Encoding) to find the most frequent character-pair patterns. Rare words split into subwords ("unhappiness" → "un", "happi", "ness"). This is why the vocab is 262K but can represent any language.

### Suggested Visual Aid
Draw a 5-word sentence as token IDs → show the tokenization process. Then show a single ID (e.g., 42) indexing into a 768-dimensional vector.

### What Students Walk Away With
- Every word = integer ID → dense vector
- Larger vocab = more compact sequences
- Embedding scale matters: too small → vanishing gradients in attention

---

## ──────────────────────────────────────────────
## Notebook 1: The Math of Attention with QK-Norm (8 min)
### ──────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Library analogy (Q/K/V) | 2 min |
| Softmax + visualization demo | 2 min |
| QK-Norm math + comparison | 3 min |
| Student exercise | 2 min |

### Teaching Points
- **The classic analogy:** Query ≈ Google search box, Key ≈ book catalog, Value ≈ book content. *Use it, but then move past it quickly.*
- **Attention formula (standard):** `softmax(Q @ K^T / sqrt(d_k)) @ V.` The `sqrt(d_k)` scaling prevents dot products from growing too large as head dimension grows.
- **Gemma 3's QK-Norm innovation:** Instead of soft-capping scores with `tanh(scores / 50.0)` (Gemma 2's approach), Gemma 3 L2-normalizes Q and K *before* the dot product. Their dot product is exactly cosine similarity — bounded in [-1, 1]. No arbitrary 50.0 constant needed.
  ```
  Q_k_norm = Q_k / ||Q_k||     (L2 norm along head_dim)
  K_k_norm = K_k / ||K_k||
  scores = Q_k_norm @ K_k_norm^T   → values in [-1, 1]
  ```
- **Output logit soft-capping (`tanh(x / 30.0)`):** Still used *after* the final LM head, NOT in attention. Purpose: prevents the model from outputting extreme confidence (probability very close to 1.0), which would cause vanishing gradients during training. 30.0 is a gentle cap — at x=30, tanh ≈ 0.995.

### 🚩 Common Misconception #1
> *"L2-normalization throws away information."* — **Correct it:** Normalization changes *scale*, not *direction*. The direction of the vector (which features are active) is preserved. Normalization only removes magnitude, which is less informative than direction for attention.

### 🚩 Common Misconception #2
> *"Softmax already limits values to [0, 1], so scores don't need capping."* — **Correct it:** The attention *weights* are in [0, 1] but the *scores* fed into softmax can be arbitrarily large. Without normalization, large scores make softmax peaky (all mass on one token), killing gradient signals elsewhere.

### Suggested Visual Aid
- **Heatmap of attention weights:** Show a clear diagonal (each token attending to itself) plus off-diagonal attention. This is the "aha" moment for students.
- **Before/after normalization:** Plot raw scores vs. QK-Norm scores on the same axes. Raw scores can span [-100, 100]; QK-Norm scores span [-1, 1].

### What Students Walk Away With
- Attention scores = dot product of Q and K → cosine similarity when normalized
- Output capping is separate from attention normalization
- QK-Norm is more stable because it has no hyperparameter to tune

---

## ──────────────────────────────────────────────
## Notebook 2: Causal Masking (5 min)
### ──────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Causal mask concept | 2 min |
| Local vs global layers | 3 min |

### Teaching Points
- **Causal mask:** Upper-triangular matrix of zeros (masked positions) + ones (visible positions). Prevents looking at future tokens.
  ```
  [[1 0 0 0]   ← token 0 can only see itself
   [1 1 0 0]   ← token 1 can see tokens 0,1
   [1 1 1 0]   ← token 2 can see tokens 0,1,2
   [1 1 1 1]]  ← token 3 can see everything before it
  ```
- **Gemma 3's hybrid attention pattern:** Gemma 3 uses **5 local (sliding window) layers** followed by **1 global layer**, repeating:
  - **Local layers:** Only attend to a sliding window of 1024 tokens before the current position. Fast + tiny KV cache.
  - **Global layer:** Attends to the *entire* sequence. Captures long-range dependencies.
  - **Ratio:** 5 local + 1 global = 15% of layers are globally attentive. This dramatically reduces overall KV-cache memory.

### 🚩 Common Misconception
> *"All GPT-style models use the same masking."* — **Correct it:** Llama uses full causal attention everywhere. Gemma 3 introduces this hybrid pattern specifically to reduce inference memory. Not all models do this.

### Suggested Visual Aid
Draw a 6-layer architecture. Color-code layers 0–4 in blue (sliding window), layer 5 in red (full). Repeat this pattern across all 8 layers of the model.

### What Students Walk Away With
- Causal masking = autoregressive generation requires it at training time
- 5:1 local/global pattern is Gemma 3's innovation for inference efficiency
- Not every model uses sliding window — Llama doesn't, Gemma does

---

## ──────────────────────────────────────────────
## Notebook 3: Grouped Query Attention (8 min)
### ──────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| MHA/MQA/GQA comparison table | 2 min |
| Repetition trick demo | 2 min |
| GQA + QK-Norm exercise | 3 min |
| Why 2 groups specifically? | 1 min |

### Teaching Points
- **MHA (Multi-Head Attention):** 8Q : 8K : 8V. Each query head has its own key/value head. Max representational quality, max KV-cache memory.
- **MQA (Multi-Query Attention):** 8Q : 1K : 1V. Single key/value head shared by all queries. Max speed, but quality degrades because all query heads attend to the same representation.
- **GQA (Grouped Query Attention):** 8Q : 2K : 2V. 2 key/value heads, each shared by 4 query heads. *Sweet spot:* near-MHA quality with 25% of the KV-cache memory.
  - Gemma 3 uses exactly 2 KV groups. Why 2 and not 1 or 4?
    - 1 group (MQA): too much sharing — head specialization is lost, attention maps become correlated
    - 4 groups: not enough saving — only 50% memory reduction, marginal speedup
    - 2 groups: enough independence for meaningful attention maps, significant memory savings

- **Implementation: the repetition trick**
  ```python
  # K has shape [batch, 2, seq_len, head_dim]  — only 2 groups!
  k_expanded = k.repeat_interleave(4, dim=1)   # → [batch, 8, seq_len, head_dim]
  # Now K matches Q's 8 heads
  scores = Q_norm @ k_expanded.norm^T  # standard attention
  ```

- **KV-cache savings with combined patterns:**
  - GQA alone: 25% of MHA memory
  - 5:1 local/global pattern: only 1/6 of layers are global (with global layers having the full cache)
  - Combined: roughly 15% of total inference memory for the cache (vs 60%+ for all-global MHA)

### 🚩 Common Misconception #1
> *"GQA sacrifices quality for speed."* — **Correct it:** GQA preserves *most* quality. The Gemma 3 technical report notes that 2 KV groups produces near-MHA accuracy. MQA (1 group) is where quality actually drops measurably.

### 🚩 Common Misconception #2
> *"More KV groups always means better quality."* — **Correct it:** Quality gains diminish sharply. Going from 8→4 groups helps slightly; 4→2 helps a little; 2→1 (MQA) hurts quality noticeably. 2 is the practical lower bound.

### Suggested Visual Aid
Draw 8 query heads as circles, and group them into 2 boxes (KV groups). Draw arrows from each Q head to its assigned KV group. This makes the sharing relationship immediately visible.

### What Students Walk Away With
- GQA is the right balance of quality/speed for Gemma 3
- `repeat_interleave` is the key implementation detail
- KV-cache savings are multiplicative with local/global pattern

---

## ──────────────────────────────────────────────
## Notebook 4: Rotary Positional Embeddings (8 min)
### ──────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| The position problem | 2 min |
| Sinusoidal vs Rotary comparison | 3 min |
| Dual-frequency RoPE | 3 min |

### Teaching Points
- **The problem:** Transformers process all tokens in parallel. Unlike RNNs, they have no inherent notion of "first token, second token, etc." Every token is permutation-invariant.
- **Solution (RoPE):** Encode position by *rotating* each pair of dimensions in the vector. Position 3 → rotate by 3×θ radians. Position 7 → rotate by 7×θ radians.
  ```
  For each (2D) rotation pair:
    x'₁ = x₁ · cos(p·θ) - x₂ · sin(p·θ)
    x'₂ = x₁ · sin(p·θ) + x₂ · cos(p·θ)
  where p = token position, θ = base angle
  ```
- **Why not sinusoidal (like original Transformer)?** Sinusoidal PE uses fixed lookup tables for each position. RoPE is *relative* — the dot product between Q at position i and K at position j depends *only* on (i − j), the relative distance. This is critical for generalizing to sequences longer than seen during training.
- **Gemma 3's dual-frequency RoPE:** Two different rotation bases:
  - **Local layers:** θ = 10,000 (standard). Fine-grained position sensitivity for short windows (1,024 tokens).
  - **Global layers:** θ = 8,000,000 (θ = 10K × 800). Extended for 128K context via [Su et al. 2021](https://arxiv.org/abs/2104.09864)'s frequency rescaling technique.
- **Why dual frequencies?** Different layer types "see" different context lengths. Local layers at window=1024 don't need extremely fine angular resolution — standard 10K works. Global layers see 128K tokens, so they need much smaller angular steps (higher θ) to distinguish positions far apart.

### 🚩 Common Misconception #1
> *"Position embeddings are added to the token embeddings."* — **Correct it:** RoPE is *rotated into* the vectors, not added to them. Token identity and position are encoded in the *direction*, not the offset. This is a fundamental architectural difference from sinusoidal PE.

### 🚩 Common Misconception #2
> *"One frequency for all layers would be simpler."* — **Correct it:** Single-frequency RoPE forces a tradeoff. Low θ → good long-range resolution but fine-grained positions blur. High θ → fine-grained positions are clear but long-range distances map to nearly zero rotation. Dual frequencies let each layer type optimize for its actual context window.

### Suggested Visual Aid
Draw two 2D vectors at positions p=0 (along X-axis) and p=3 (rotated 3θ). Show how the angle between them = 3θ regardless of which token vectors we use (relative encoding). Then overlay another pair at different absolute positions but the same relative distance — same angle → same attention weight.

### What Students Walk Away With
- Position encoding turns "bag of words" into ordered sequence
- RoPE encodes *relative* position, enabling length generalization
- Dual frequencies let local and global layers each use what works for their context window

---

## ──────────────────────────────────────────────
## Recap: What We've Built So Far (2 min)
### ──────────────────────────────────────────────

**Narrative summary for the class:**
> "We've built the *vision* of the model. Here's the flow:
> 1. Input text → token IDs → 768-dim vectors (×√768 for scaling)
> 2. QK-Norm: Q and K vectors are L2-normalized → dot product = cosine similarity
> 3. Causal mask: upper-triangular matrix prevents looking ahead
> 4. GQA: 2 KV heads shared by 4 query heads each → 25% KV-cache memory
> 5. RoPE: position encoded via rotation → relative distance = attention weight
> 
> All of this is the *same* attention mechanism, but each optimization makes it faster or more stable."

---

## ──────────────────────────────────────────────
## Bridge to Next Block (1 min)
### ──────────────────────────────────────────────

> "Attention gives the model its *vision*. But vision alone isn't enough — the model also needs to *think*. In the next block (notebooks 5–7), we add the thinking machinery: gating, normalization, and the full transformer block."
