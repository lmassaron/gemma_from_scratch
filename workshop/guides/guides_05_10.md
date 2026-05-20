# Presentation Guide: Notebooks 05–10 (Gated MLP → RMSNorm → Transformer Block → Assembly → Inference → Training)

---

## 🕐 Total Time for This Block: ~75 minutes
## 🧱 Building the "Thinking" of the Model

> **Presenter Note:** This block assembles the *transformer architecture*. These are the hardest notebooks conceptually. Slow down here. Emphasize connections between notebooks — students should see each notebook completing a puzzle piece, not learning a disconnected topic.

---

## ──────────────────────────────────────────────────
## Notebook 5: Gated MLP (GeGLU) (8 min)
### ────────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| MLP vs gated MLP motivation | 2 min |
| GeGLU vs SwiGLU architecture comparison | 3 min |
| Exercise: implement SimpleGatedMLP | 3 min |

### Teaching Points
- **Standard MLP:** `output = W2 @ GELU(W1 @ input)` — one intermediate layer with GELU activation.
- **Gated MLP (Gemma 3's choice):** Three projections instead of two:
  ```
  gate_output = W_gate @ input          # projects to intermediate_size
  up_output   = W_up @ input            # projects to intermediate_size
  gated       = GELU(gate_output) * up_output  # element-wise multiplication
  output      = W_down @ gated          # back to hidden_size
  ```
- **Why gating?** Gating is *adaptive*: the gate vector changes per input token. For one token, it might suppress certain features; for another, it amplifies different features. It's not a static nonlinear transformation — it's a *per-sample filter*.
- **GeGLU vs SwiGLU (architecture details):**
  - **GeGLU (Gemma 3):** Gates with GELU: `GELU(gate) * up`
  - **SwiGLU (Llama 3):** Gates with SiLU: `SiLU(gate) * up`
  - Both are gated MLPs. The original GeGLU paper ([Shazeer 2020](https://arxiv.org/abs/2002.05202)) found that GELU gating outperformed simple GELU on language tasks. Gemma 3 chose GeGLU. Llama 3 chose SwiGLU — both work well; the difference is minor and data-dependent.
- **intermediate_size > hidden_size:** Gemma 3's MLP expands dimensions (768 → 2048 → 768). Why? More parameters in the intermediate space → more capacity to learn complex transformations. This adds ~3× the parameters of the embedding alone.

### 🚩 Common Misconception #1
> *"Gating is just an activation function like ReLU."* — **Correct it:** A gating mechanism has a *learned* gate. ReLU is fixed (0 or x, no parameters). Gating is a *learned per-token filter* — the model learns what to keep and what to suppress for each specific input.

### 🚩 Common Misconception #2
> *"GeGLU is superior to SwiGLU."* — **Correct it:** It's a data-dependent choice. Both architectures perform similarly across benchmarks. Gemma 3 uses GeGLU; Llama 3 uses SwiGLU. Neither is universally better — the "best" choice depends on the training data and optimization recipe.

### Suggested Visual Aid
Draw the three-branch MLP: gate_proj and up_proj both take input, their outputs meet at the multiplication node, GELU is applied to the gate branch, then the product goes to down_proj. Color-code the gate branch in green (dynamic filter) and up branch in blue (features).

### What Students Walk Away With
- Gated MLPs have three projections, not two
- Gating is learned and per-token (adaptive, not static)
- intermediate_size > hidden_size (expansion for capacity)
- GeGLU and SwiGLU are comparable; Gemma 3 happens to use GeGLU

---

## ──────────────────────────────────────────────────
## Notebook 6: RMSNorm & Double Normalization (7 min)
### ────────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| RMSNorm vs LayerNorm | 2 min |
| Double-norm pattern | 2 min |
| Add-one trick | 2 min |
| Verify the identity initialization | 1 min |

### Teaching Points
- **Why not LayerNorm?** LayerNorm computes: `((x - mean(x)) / std(x)) * weight + bias`. RMSNorm removes the mean subtraction and bias: `x / RMS(x) * weight + bias`. Simpler, faster (fewer operations), and empirically nearly identical in performance.
  - RMSNorm formula: `x / sqrt(mean(x^2) + epsilon) * gamma`
  - RMSNorm is just *normalizing to unit root-mean-square* — no mean subtraction.
- **Double-normalization in Gemma:** Every sub-layer (attention and MLP) has normalizations at *both* input and output positions. This is a Gemma signature:
  ```
  # Attention sub-layer
  h1 = RMSNorm(x)                    # Pre-norm: normalize BEFORE attention
  h2 = h1 + Attention(h1)            # Residual: add original input
  h3 = RMSNorm(h2)                   # Post-norm: normalize AFTER attention
  
  # MLP sub-layer
  h4 = RMSNorm(h3)                   # Pre-norm: normalize BEFORE MLP
  h5 = h4 + MLP(h4)                  # Residual: add
  output = h5                        # Final double-normed output
  ```
  Note: **This is NOT the same as LayerNorm in Llama.** Llama uses *only* pre-norm (one RMSNorm per sub-layer, at the start). Gemma adds a *second* RMSNorm after each sub-layer for extra stability. This is the "double-norm" pattern.
- **Add-one trick:** The RMSNorm weights are initialized to **zero**. The scaling formula is `(1 + weight)`, which means:
  - At initialization: `(1 + 0) = 1` → the layer is an **exact pass-through (identity)**.
  - As training progresses, the weights grow from zero, letting the normalization strength adapt.
  - This is more stable than initializing to 1.0 because it starts as identity and *learns* the normalization.

### 🚩 Common Misconception #1
> *"RMSNorm and LayerNorm are nearly identical, so the choice doesn't matter."* — **Correct it:** While both normalize, RMSNorm is **faster to compute** (no mean subtraction) and **requires fewer gradient updates** (stable gradients). This matters in deep networks where every FLOP counts.

### 🚩 Common Misconception #2
> *"Normalizing the output of a sub-layer is unnecessary."* — **Correct it:** Gemma's double-norm (pre + post) is specifically designed for stability in deep networks. When gradients flow through 8+ layers of attention and MLP, each layer amplifies or dampens the signal. Normalizing *after* each sub-layer prevents signal explosion/diminishment, keeping the gradient scale consistent through depth.

### Suggested Visual Aid
Show the double-norm flow as a box diagram:
```
 x → [RMSNorm] → [Attention] → [RMSNorm] → [RMSNorm] → [MLP] → [RMSNorm] → output
      (pre)       (residual)   (post)      (pre)        (residual)  (post)
```
Emphasize: **4 RMSNorm calls per transformer block.** This is Gemma's signature architectural detail.

### What Students Walk Away With
- RMSNorm = LayerNorm without mean subtraction → faster, simpler
- Gemma uses 4 RMSNorm calls per block (pre-norm + post-norm for both attention and MLP)
- Weight=0 initialization + (1+weight) → layer starts as identity

---

## ──────────────────────────────────────────────────
## Notebook 7: The Transformer Block (10 min)
### ────────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Review: what's in one block | 2 min |
| Layer pattern (5:1) assembly | 3 min |
| Tensor shapes walkthrough | 3 min |
| Run and debug | 2 min |

### Teaching Points
- **What's one transformer block?** It's everything we've built combined:
  1. **QK-Norm attention:** Q and K are L2-normalized, dot product → cosine similarity
  2. **Causal mask:** upper-triangular boolean mask
  3. **GQA:** 2 KV heads shared by 4 Q heads each, expanded via `repeat_interleave`
  4. **RoPE:** dual-frequency rotation angles for positional encoding
  5. **Gated MLP:** GeGLU with 3 projections
  6. **Double-norm:** 4 RMSNorm calls per block
  7. **Residual connections:** two `x + f(x)` operations per block

- **The 5:1 layer pattern (revisited):**
  ```
  Block 1:  Local (window=1024)  — fast, 1024-token context
  Block 2:  Local (window=1024)
  Block 3:  Local (window=1024)
  Block 4:  Local (window=1024)
  Block 5:  Local (window=1024)
  Block 6:  Global (full context) — long-range awareness
  Block 7:  Local (window=1024)
  Block 8:  Global (full context — only 1 more, total model)
  ```
  In Gemma 3 (8 layers total): *every 6th layer is global*. So layers 5 and (would-be) 11 are global. Since the model only has 8 layers, only layer 5 is global. This saves KV cache because only global layers need the full cache.

- **Implementation detail: is_local gating:** The `is_local` flag switches between a sliding-window mask and a full-attention mask. Same module, different masks.

### 🚩 Common Misconception #1
> *"The sliding window means information gets lost after 1024 tokens."* — **Correct it:** Global layers act as "bridge" layers. Every 6th layer sees the full context. So information from 1024 tokens ago flows through the global layer and propagates forward through subsequent local layers. It's a *sparse long-range* path, not a complete path — efficient but not infinite-range.

### 🚩 Common Misconception #2
> *"Residual = skip connection."* — **Correct it:** A residual connection is `x + f(x)`, a skip connection is `x + W_skip(x)`. They are often used interchangeably, but in Gemma, the residual is *exactly* `x + sublayer_output` — no weight on the shortcut.

### Suggested Visual Aid
Draw a single transformer block. Color-code the double-norm path in red pre-norm, green post-norm. Show the two residual arrows (from before attention and from before MLP). This is the most detailed architectural diagram students will see, so make it clear.

### What Students Walk Away With
- One block = attention + GQA + RoPE + Gated MLP + double-norm + residuals
- Layer pattern 5:1 = 5 local + 1 global repeating
- Residuals are the "highways" connecting layers

---

## ──────────────────────────────────────────────────
## Notebook 8: Full Gemma 3 Model Assembly (10 min)
### ────────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Model anatomy walkthrough | 3 min |
| Parameter counting exercise | 3 min |
| Run greedy generation | 4 min |

### Teaching Points
- **Complete architecture:**
  ```
  Input Tokens [B, T]
    → token_embeddings [B, T, 768] × sqrt(768)   (embedding scaling)
    → transformer_blocks (8x, 5 local + 1 or 2 global)
    → final_rmsnorm
    → lm_head (weight-tied to embeddings)
    → tanh(logits / 30.0)                         (logit cap)
    → softmax → probability distribution [B, V]
  ```

- **Weight tying:** `lm_head.weight = token_embeddings.weight.T`. The output projection matrix *is* the embedding matrix. This is a standard trick: it shares ~2× vocab × hidden parameters (for Gemma 3: 262K × 768 ≈ 400M parameters) and regularizes learning.

- **Parameter budget (pedagogical ~270M model):**
  - Embeddings: 262K × 768 ≈ 200M parameters
  - Linear layers in attention (Q/K/V/out): 4 × (768 × 768) = 2.3M
  - Linear layers in MLP (gate/up/down): 3 × (768 × (768×8/3)) ≈ 4.4M (with expansion)
  - RMSNorm params (4 per block): 4 × 768 × 8 = 24.6K (tiny, scale vectors)
  - Total: ≈ 270M parameters (mostly embeddings dominate)

- **Greedy generation demo:** The model outputs the token with the *highest probability* at each step. Deterministic. Useful for debugging but not for creative tasks.

### 🚩 Common Misconception #1
> *"The model can generate any length sequence."* — **Correct it:** RoPE and causal masks are built for a **specific max position**. Exceeding it produces garbage. Gemma 3 is built for 128K tokens via RoPE scaling, but our toy model is built for 512 tokens.

### 🚩 Common Misconception #2
> *"Weight tying means the model only has half the parameters."* — **Correct it:** Weight tying means the **output projection shares parameters with input embeddings**, saving ~2 × vocab × hidden_size (but only the *weight matrix*, not the bias). It's a regularization technique, not a compression technique.

### Suggested Visual Aid
Draw the complete model as a pipeline diagram, from left to right:
```
Text → [Tokenizer] → IDs → [Embedding × √768] → [Transformer Block] → ... → [RMSNorm] → [LM Head] → [Softcap] → [Softmax] → Tokens
```
Label the Gemma 3 innovations on each stage.

### What Students Walk Away With
- Full model = embedding + scaled blocks + RMSNorm + LM head + softcap + softmax
- Weight tying = output projection = embedding matrix transposed
- Parameter count is dominated by embeddings, not attention
- Model can only handle sequences up to its max length

---

## ──────────────────────────────────────────────────
## Notebook 9: Inference & Sampling (10 min)
### ────────────────────────────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Why we need KV cache (timing demo) | 3 min |
| Implement KV cache | 3 min |
| Temperature & Top-p sampling | 3 min |
| Full generation loop | 1 min |

### Teaching Points
- **The inference problem:** For autoregressive generation, we predict token-by-token. At step t, the model needs the hidden states for *all previous* tokens to compute attention. Naive recompute at each step is O(n²) per step = O(n³) total. **KV cache reduces this to O(n²) total** by storing past K,V.
- **KV cache in action:**
  - Step 1: compute K₁,V₁ for token 1. Store them.
  - Step 2: compute K₂,V₂ for token 2. Concatenate [K₁, K₂], [V₁, V₂].
  - Step 3: compute K₃,V₃ for token 3. Concatenate [K₁, K₂, K₃], [V₁, V₂, V₃].
  - At step t: K cache has shape [batch, t, num_kv_groups, head_dim]
- **GQA with KV cache:** Since only 2 KV groups (not 8), our KV cache is 8x smaller. With the 5:1 local/global pattern, only ~2 layers need the full cache.
- **Temperature:** Controls output randomness. Lower T → conservative/more confident. Higher T → creative/diverse.
  ```
  logits_T = logits / T            # T < 1: sharpen, T > 1: soften
  probs = softmax(logits_T)
  ```
  - T = 0.1 → very deterministic (almost always pick argmax)
  - T = 1.0 → default, proportional to logit
  - T = 2.0 → very stochastic (probabilities flatten toward uniform)
- **Top-p (nucleus) sampling:** Only sample from the *top p fraction* of probabilities, then renormalize.
  ```
  probs_sorted = sort(probs, descending=True)
  cumulative = cumsum(probs_sorted)
  keep = cumulative ≤ p       # p = 0.9 → keep 90% of probability mass
  probs[~keep] = 0            # zero out the rest
  probs = probs / probs.sum() # renormalize
  ```
  - p = 0.9: most diverse, good for creative tasks
  - p = 0.7: balanced for general use
  - p = 0.5: conservative, good for factual questions
- **Combined strategy:** Apply logit cap → temperature → top-p → sample. Always.

### 🚩 Common Misconception #1
> *"KV cache just stores the output embeddings."* — **Correct it:** KV cache stores the **Key and Value matrices from every attention sub-layer at every time step.** It's not embeddings — it's the internal K and V states that attention uses to compute dot products.

### 🚩 Common Misconception #2
> *"Temperature controls how confident the model is *in* its answer."* — **Correct it:** Temperature controls how *randomly* we select from the model's probability distribution. The model's confidence comes from the logit values, not from temperature. Temperature is a *sampling knob*, not a calibration tool.

### Suggested Visual Aid
- **KV cache timing demo:** Show generation at step 100 *without* KV cache (time it: ~5 seconds), then *with* KV cache (time it: ~5ms). The speed difference is dramatic and memorable.
- **Temperature visualization:** Take one logit vector, show the softmax at T=0.1, T=1.0, T=5.0. The distributions go from a single spike → moderate spread → nearly flat. Students *see* what temperature does.
- **Top-p visualization:** Same logit distribution. Sort, cumsum, show the "keystone" — the point where cumulative probability reaches p. Mark: "everything to the left of this point survives."

### What Students Walk Away With
- KV cache = store past K,V → recompute once, reuse forever
- Temperature = randomness knob (not confidence)
- Top-p = sample from p% highest-probability tokens
- Generation strategy: logit cap → temperature → top-p → sample

---

## ──────────────────────────────────────────────────
## Notebook 10: Training Loop (10 min)
### ──────────────────────---─────────────────────────

### Timing Guide
| Phase | Duration |
|-------|---------|
| Training loop overview | 2 min |
| Forward pass, loss, backward | 4 min |
| Training demo: watch loss change | 4 min |

### Teaching Points
- **The training loop:** What makes a language model *learn*?
  ```
  for epoch in range(num_epochs):
      for batch_text in dataloader:
          logits = model(batch_text)                           # forward pass
          loss = cross_entropy(logits_true, batch_target)      # compute loss
          loss.backward()                                        # compute gradients
          optimizer.step()                                       # update weights
          optimizer.zero_grad()
  ```
  This is **the entire training loop** of Gemma 3 (and GPT-4 and every other LLM). The mechanism is the same at every scale. The difference is just:
  - Data volume: 5 examples (our workshop) vs. trillions of tokens (Gemma 3)
  - Sequence length: 512 (our toy) vs. up to 128K (Gemma 3)
  - Batch size: 2 (our toy) vs. 8192+ (Gemma 3 on TPU clusters)
  - Hardware: single GPU (our workshop) vs. TPU v5p clusters with thousands of chips

- **Cross-entropy loss:** Measures how wrong the model's probability distribution is compared to the true next token.
  ```
  loss = -log(proportion of correct token's probability)
  ```
  Example: if the true token has probability 0.01, loss = -log(0.01) ≈ 4.6. If probability = 0.99, loss ≈ 0.01.

- **Gradient clipping:** Essential for stable training. Without it, exploding gradients (common in LLMs due to chain rule through many layers) cause NaN loss.
  ```
  total_norm = gradient_l2_norm()
  if total_norm > max_norm:
      scale = max_norm / total_norm
      gradients = gradients * scale
  ```
  Gemma 3 uses max_norm=1.0.

- **The training journey (what to expect):**
  - **Step 0 (random):** Loss ≈ log(vocab) ≈ log(262K) ≈ 12.5. The model predicts uniformly at random.
  - **Step 100-500:** Loss drops rapidly as the model learns trivial patterns (punctuation, spacing).
  - **Step 500-1,000:** Loss plateaus. The model can do *something* with short sequences but not general patterns.
  - **Step 1,000+:** With enough data and steps, the model starts *generating anything that looks like language*. Our toy dataset of 5 examples will stop improving after ~1,000 steps — this is expected.

- **What real Gemma 3 training does (for context):**
  - **Pre-training:** ~2 trillion tokens on a massive corpus (multilingual text, code, math, reasoning)
  - **Optimization:** AdamW with cosine learning rate schedule
  - **Infrastructure:** TPUs (not GPX — TPU clusters with ZeRO-3 sharding across thousands of TPUs)
  - **Loss scaling:** Multi-stage — starts with high learning rate, decays to 0 via cosine schedule
  - **Post-training:** Supervised fine-tuning (SFT) on high-quality conversational examples, followed by Direct Preference Optimization (DPO) for alignment
  - **Safety:** Red-teaming, harmful content filtering, bias detection

### 🚩 Common Misconception #1
> "Training a model is just about having more data." — **Correct it:** Training quality depends on *data quality*, not just quantity. Gemma 3 was trained on 2 trillion tokens, but 90%+ of the value came from careful data curation, deduplication, and safety filtering — not raw volume.

### 🚩 Common Misconception #2
> "Once the model is trained, it's 'done'." — **Correct it:** Real-world models go through multiple training *stages*:
  1. **Pre-training:** Learn language patterns from raw text
  2. **Instruction tuning:** Learn to follow instructions (SFT)
  3. **Preference alignment:** Learn to produce helpful, safe responses (DPO)
  4. **Continued pre-training:** Continuously update on new data
  Each stage produces a different model. Gemma 3's *final* model is the result of all stages, not just the initial pre-training.

### Suggested Visual Aid
- **Loss curve:** Plot loss vs. step on a log scale. Show the steep initial drop → plateau → gradual decline. This is the most important visualization in the workshop.
- **Training vs. inference comparison table:**
  | Aspect | Training | Inference |
  |--------|----------|-------------|
  | Input | Entire batch of text | One token at a time |
  | Target | Next token for every position | One token (user prompt) |
  | Loss | Yes (cross-entropy) | No |
  | Gradient | Yes (backward pass) | No |
  | Speed | Slow (backward + optimizer) | Fast (KV cache) |
  | Memory | High (stores all activations) | Moderate (stores K,V cache) |

### What Students Walk Away With
- The training loop is simple: forward → loss → backward → step
- Loss ≈ log(vocab) at random start → decreases as model learns
- Gradient clipping prevents NaN explosions
- Real training = many stages (pre-training → SFT → alignment → more training)

---

## ──────────────────────────────────────────────────
## Final Recap & Architecture Summary (5 min)
### ──────────────────────---─────────────────────

**Show this complete architecture on screen:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                     GEMMA 3 ARCHITECTURE SUMMARY                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Tokenization                                                     │
│     Text → BPE (262K tokens) → IDs → Embeddings (768×) × √768      │
│                                                                      │
│  2. Per-Layer Pattern (5 local + 1 global, repeating)                │
│     x ─→ RMSNorm ─→ Attention (QK-Norm + GQA + RoPE + Mask)        │
│                      ─→ RMSNorm ─→ [Add x] ─→ RMSNorm ─→             │
│                      GeGLU(gate×up) ─→ out ─→ RMSNorm ─→ [Add]       │
│                                                                      │
│  3. Output                                                           │
│     hidden ─→ RMSNorm ─→ LM Head (weight-tied) ─→ tanh(x/30)        │
│                      ─→ softmax ─→ Next Token                         │
│                                                                      │
│  4. Inference                                                        │
│     KV Cache (8Q:2KV GQA + 5:1 layers → <15% memory)                │
│     Temperature (randomness) + Top-p (nucleus sampling)              │
│                                                                      │
│  ALL Gemma 3 Innovations in This Workshop:                           │
│  ✅ QK-Norm (cosine attention)                                     │
│  ✅ GQA 8Q:2KV (grouped query attention)                            │
│  ✅ Dual RoPE (10K local / 8M global)                               │
│  ✅ GeGLU gating (3-projection MLP)                                  │
│  ✅ Double RMSNorm (4 calls per block)                              │
│  ✅ Logit cap tanh(x/30)                                            │
│  ✅ Weight tying (lm_head = embeddings)                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ──────────────────────────────────────────────────
## Workshop Close (3 min)
### ──────────────────────---─────────────────────

- **What we've built:** A fully functional 270M-parameter Gemma 3 decoder from scratch in ~11 notebooks (no HF libraries, no shortcuts).
- **What's next for learners:**
  1. Use your model to generate text with different temperatures (notebook 9).
  2. Try training on a slightly larger dataset and observe convergence (notebook 10).
  3. Explore how Gemma 3 handles multilingual text and code.
  4. Compare your implementation against `transformers.AutoModelForCausalLM` from the HuggingFace library.

- **Key takeaway:** Every large language model — GPT, Gemma, Llama, Claude — is built on this same architecture. The differences are in data, scaling, and optimization. The *math* is the same. You now understand it.
