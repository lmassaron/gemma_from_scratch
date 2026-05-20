# Gemma 3 From Scratch — PyCon Workshop Study Notes

> **Workshop:** Building Gemma 3 from Scratch in PyTorch  
> **Scope:** ~270M-parameter educational clone of Google's Gemma 3 architecture  
> **Notebooks:** 11 notebooks (00–10), estimated ~2 hours total

---

## Quick-Reference: Architecture at a Glance

| Hyperparameter | Value | Notes |
|---|---|---|
| `vocab_size` | 256,000 | SentencePiece BPE, same as Gemini 2.0 |
| `hidden_size` | 768 | Embedding width |
| `num_layers` | 8 | Transformer blocks |
| `num_attention_heads` | 8 | Query heads |
| `num_kv_heads` | 2 | KV heads (GQA) |
| `head_dim` | 96 | = 768 / 8 |
| `intermediate_size` | 2048 | GeGLU hidden dim |
| `sliding_window` | 1024 | Local layer window |
| `rope_theta_local` | 10,000 | RoPE for local layers |
| `rope_theta_global` | 1,000,000 × 8 | RoPE for global layers (128K ctx) |
| `logit_cap` | 30.0 | Final output tanh cap |
| `hidden_scale` | √768 ≈ 27.7 | Embedding multiplier |

**Official Gemma 3 sizes:** 1B, 4B, 12B, 27B. The 270M variant is a workshop toy scale only.

---

## Part 1 — Data Pipeline (Notebook 00)

### 1.1 Tokenization

Gemma 3 uses a **SentencePiece BPE tokenizer** with 256K vocabulary. Properties:
- Byte-level (handles any language and rare characters)
- Digits split separately (`"123"` → separate tokens)
- Preserves whitespace (important for code)

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b")
ids = tokenizer.encode("Hello world")
```

Special tokens: `<pad>=0`, `<eos>=1`, `<bos>=2`, `<unk>=3`

### 1.2 Embedding Layer

```python
tok_emb = nn.Embedding(vocab_size, hidden_size)
# Input:  token IDs  → shape (B, T)
# Output: vectors    → shape (B, T, hidden_size)
```

**Weight tying** — the embedding matrix and the final LM head share the same weights:
```python
self.lm_head.weight = self.tok_embeddings.weight  # saves ~400M params
```

### 1.3 Embedding Scaling (Gemma's Trick)

Embeddings are scaled by `√hidden_size` immediately after lookup:
```python
hidden_scale = hidden_size ** 0.5   # ≈ 27.7
x = tok_emb(input_ids) * hidden_scale
```

**Why?** Embedding weights initialise with small variance. The scale factor ensures the signal entering the first layer has reasonable magnitude, preventing vanishing gradients.

---

## Part 2 — Attention Mechanics (Notebooks 01–03)

### 2.1 QK-Norm (Gemma 3's Core Innovation)

**Gemma 2** used attention score soft-capping: `tanh(scores / 50.0)`.  
**Gemma 3** replaces this with **L2 normalisation of Q and K** before the dot product.

```python
Q_norm = F.normalize(Q, p=2, dim=-1)   # unit vectors
K_norm = F.normalize(K, p=2, dim=-1)

scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / math.sqrt(head_dim)
```

The dot product of two unit vectors is cosine similarity, bounded in `[-1, 1]`, making scores inherently stable. We still divide by `√d_k` to prevent very small gradients when `head_dim` is large.

Full QK-Norm attention:
```python
def qk_attention(Q, K, V):
    d_k = Q.shape[-1]
    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)
    scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

### 2.2 Final Logit Soft-Capping (Separate from Attention)

After the full transformer, the *output* logits are capped (not the attention scores):
```python
logit_cap = 30.0
capped_logits = logit_cap * torch.tanh(logits / logit_cap)
```

**Why cap at 30.0?** Prevents the output softmax from becoming too peaked (probability ≈ 1.0 on one token), which causes vanishing gradients for all other tokens and destabilises training.

### 2.3 Causal Masking (Notebook 02)

Decoder-only models must not see future tokens. The mask sets future positions to `-inf` before softmax:

```python
# Global causal mask
ones = torch.ones((seq_len, seq_len), dtype=torch.bool)
mask_global = torch.triu(ones, diagonal=1)            # upper triangle = future

# Apply to scores
scores = scores.masked_fill(mask_global, float('-inf'))
weights = torch.softmax(scores, dim=-1)               # future positions → 0.0
```

**Sliding window (local) mask** — additionally masks tokens too far in the past:
```python
one_row = torch.ones((1, seq_len), dtype=torch.bool)
far_past = torch.triu(one_row, diagonal=sliding_window).expand(seq_len, -1).T
mask_local = mask_global | far_past
```

**5:1 local/global layer pattern:**

| Layer index | Type | Attention window | RoPE θ |
|---|---|---|---|
| 0, 1, 2, 3, 4 | LOCAL | 1024 tokens | 10K |
| 5 | GLOBAL | Full context | 8M |
| 6, 7, 8, 9, 10 | LOCAL | 1024 tokens | 10K |
| 11 | GLOBAL | Full context | 8M |

```python
def is_global_layer(layer_idx):
    return layer_idx % 6 == 5    # every 6th layer (0-indexed)
```

### 2.4 Grouped Query Attention — GQA (Notebook 03)

| Attention type | Q heads | K heads | V heads | KV-cache vs MHA |
|---|---|---|---|---|
| MHA | 8 | 8 | 8 | 1× |
| MQA | 8 | 1 | 1 | 0.125× |
| **GQA (Gemma 3)** | **8** | **2** | **2** | **0.25×** |

K and V are projected to fewer heads and then expanded to match Q:
```python
self.W_q = nn.Linear(d_in, n_heads * head_dim, bias=False)        # 8 heads
self.W_k = nn.Linear(d_in, n_kv_groups * head_dim, bias=False)    # 2 heads
self.W_v = nn.Linear(d_in, n_kv_groups * head_dim, bias=False)    # 2 heads

# Expand K, V to match Q
group_size = n_heads // n_kv_groups   # = 4
k = k.repeat_interleave(group_size, dim=1)   # (B, 2, T, D) → (B, 8, T, D)
v = v.repeat_interleave(group_size, dim=1)
```

Full projection flow:
```python
def forward(self, x):
    B, T, C = x.shape
    q = self.W_q(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = self.W_k(x).view(B, T, n_kv_groups, head_dim).transpose(1, 2)
    v = self.W_v(x).view(B, T, n_kv_groups, head_dim).transpose(1, 2)
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    # ... QK-Norm attention ...
    out = out.transpose(1, 2).contiguous().view(B, T, n_heads * head_dim)
    return self.out_proj(out)
```

---

## Part 3 — Positional Embeddings (Notebook 04)

### 3.1 RoPE — Rotary Positional Embeddings

Transformers are permutation-invariant: they need explicit positional information. RoPE encodes position as a **rotation** of Q and K vectors in 2D subspaces.

**Key property:** The dot product of a rotated Q at position `m` and rotated K at position `n` depends only on the *relative distance* `(m - n)`, not absolute positions.

**Step 1 — Compute inverse frequencies:**
```python
inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

**Step 2 — Compute angle per position:**
```python
t = torch.arange(seq_len)
freqs = torch.outer(t, inv_freq)    # (seq_len, head_dim/2)
cos_freqs = torch.cos(freqs)
sin_freqs = torch.sin(freqs)
```

**Step 3 — Apply rotation:**
```python
def apply_rope(x, cos, sin):
    x_left  = x[..., :head_dim // 2]
    x_right = x[..., head_dim // 2:]
    x_rotated = torch.cat([-x_right, x_left], dim=-1)
    return (x * cos) + (x_rotated * sin)
```

### 3.2 Dual-Frequency RoPE (Gemma 3)

| Layer type | θ base | Effect |
|---|---|---|
| Local (0-4, 6-10…) | 10,000 | Standard short-range rotation |
| Global (5, 11…) | 1,000,000 × 8 = 8,000,000 | Much slower rotation; encodes long-range distances |

Higher θ → slower rotation per position → finer-grained relative position encoding over long sequences. This enables 128K context in global layers.

```python
def get_rope_theta(layer_idx):
    if layer_idx % 6 == 5:
        return 1_000_000.0 * 8.0   # global
    return 10_000.0                 # local
```

---

## Part 4 — Feed-Forward Network (Notebook 05)

### 4.1 GeGLU vs SwiGLU

| Variant | Gate activation | Used in |
|---|---|---|
| SwiGLU | `SiLU(gate) × up` | Llama, Gemma 2 |
| **GeGLU** | **`GELU(gate) × up`** | **Gemma 3** |

The architecture uses **three projections**:

```python
class Gemma3GeGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.up_proj   = nn.Linear(d_in, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        activated_gate = F.gelu(gate, approximate='tanh')   # GELU, tanh approx for speed
        return self.down_proj(activated_gate * up)          # element-wise multiply
```

Formula: `Output = W_down( GELU(W_gate · x)  ⊙  (W_up · x) )`

**Why gating?** The gate acts as a per-token, per-feature filter. It learns to dynamically suppress or amplify features from the up projection — more expressive than a plain 2-layer MLP.

**Parameter count:** GeGLU uses 3 matrices (gate + up + down), vs 2 for a vanilla MLP — roughly 50% more parameters at the same `intermediate_size`. This is intentional; the gate capacity is worth the cost.

---

## Part 5 — Normalisation (Notebook 06)

### 5.1 RMSNorm vs LayerNorm

| Method | Formula | Notes |
|---|---|---|
| LayerNorm | `(x - E[x]) / sqrt(Var[x] + ε) × γ + β` | Subtracts mean, two learnable params |
| **RMSNorm** | **`x / sqrt(mean(x²) + ε) × γ`** | No mean subtraction, one learnable param |

RMSNorm is faster and simpler. Skipping the mean subtraction works well in practice.

### 5.2 Gemma's Add-One Trick

```python
class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))   # init at ZERO, not one

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * (1.0 + self.weight)
                                                #  ↑ add-one: (1 + 0) = 1 at init
```

At initialisation `weight=0`, so the scale factor is exactly `1.0`. The layer starts as a pure RMS-normaliser — an identity-like operation. This prevents skewed gradients from random scaling on the very first steps.

### 5.3 Double Normalisation (Gemma 3 Signature)

Every sub-layer (attention AND MLP) gets **both** pre-norm and post-norm:

```python
# For both attention and MLP sub-layers:
shortcut = x
x = RMSNorm_pre(x)       # pre-norm
x = SubLayer(x)          # attention or MLP
x = RMSNorm_post(x)      # post-norm  ← Gemma's unique addition
x = shortcut + x         # residual
```

This requires 4 RMSNorm instances per block (2 for attention, 2 for MLP).

| Architecture | Norm placement | Stability |
|---|---|---|
| Post-norm (original) | After each sub-layer | Unstable at depth |
| Pre-norm (Llama, T5) | Before each sub-layer | Moderate |
| **Double-norm (Gemma 3)** | **Pre + Post on every sub-layer** | **Most stable** |

---

## Part 6 — Transformer Block Assembly (Notebook 07)

```python
class Gemma3TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_groups, h_dim, hidden_dim, is_local=True):
        super().__init__()
        self.attn = Gemma3GQA(dim, n_heads, n_kv_groups, h_dim, is_local)
        self.ffn  = Gemma3GeGLU(dim, hidden_dim)
        # 4 norms: 2 for attention, 2 for MLP
        self.input_layernorm          = Gemma3RMSNorm(dim)
        self.post_attn_layernorm      = Gemma3RMSNorm(dim)
        self.pre_feedforward_layernorm  = Gemma3RMSNorm(dim)
        self.post_feedforward_layernorm = Gemma3RMSNorm(dim)

    def forward(self, x):
        # Attention sub-block
        shortcut = x
        x = self.input_layernorm(x)
        x = self.attn(x)
        x = self.post_attn_layernorm(x)
        x = shortcut + x

        # MLP sub-block
        shortcut = x
        x = self.pre_feedforward_layernorm(x)
        x = self.ffn(x)
        x = self.post_feedforward_layernorm(x)
        x = shortcut + x
        return x
```

The block preserves shape: input `(B, T, hidden_size)` → output `(B, T, hidden_size)`.

**Parameter count per block (approximate):**
- Attention (Q, K, V, out): `4 × hidden_size²`
- GeGLU MLP (gate, up, down): `3 × hidden_size × intermediate_size`
- 4 × RMSNorm weights: `4 × hidden_size`

---

## Part 7 — Full Model Assembly (Notebook 08)

### 7.1 Complete Architecture

```
Input IDs (B, T)
      ↓
Embedding(256K, 768)  × sqrt(768)
      ↓
┌─ Transformer Blocks × 8 ──────────────────────────┐
│  Layer 0-4: LOCAL  (sliding_window=1024, θ=10K)   │
│  Layer 5:   GLOBAL (full attention, θ=8M)          │
│  Layer 6-7: LOCAL  (sliding_window=1024, θ=10K)   │
│                                                    │
│  Each block: GQA + GeGLU + 4×RMSNorm              │
└────────────────────────────────────────────────────┘
      ↓
Final RMSNorm
      ↓
LM Head  768 → 256K  (weights tied with embedding)
      ↓
tanh(logits / 30.0) × 30.0   ← logit cap
      ↓
Logits (B, T, 256K)
```

### 7.2 Model Class Structure

```python
class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma3TransformerBlock(
                dim=config.hidden_size,
                n_heads=config.num_heads,
                n_kv_groups=config.num_kv_heads,
                h_dim=config.head_dim,
                hidden_dim=config.intermediate_size,
                is_local=(i % 6 != 5),
                sliding_window=config.sliding_window if (i % 6 != 5) else None
            )
            for i in range(config.num_layers)
        ])
        self.norm   = Gemma3RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids) * hidden_scale
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logit_cap * torch.tanh(logits / logit_cap)
```

### 7.3 Greedy Decoding

```python
def generate(model, prompt_ids, max_length=30):
    model.eval()
    input_ids = prompt_ids.clone()
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == 1:   # <eos>
            break
    return input_ids[0]
```

---

## Part 8 — Inference & Sampling (Notebook 09)

### 8.1 KV Cache

In autoregressive generation, a naive approach recomputes all past K and V at every step — O(n²) per layer. **KV cache** stores past K,V vectors and only computes the new token's — O(n).

```python
class KVCache:
    def __init__(self, n_kv_groups, h_dim, max_cache_len=512):
        self.keys   = torch.zeros(1, n_kv_groups, max_cache_len, h_dim)
        self.values = torch.zeros(1, n_kv_groups, max_cache_len, h_dim)
        self.offset = 0

    def update(self, k, v):
        T = k.shape[2]
        self.keys  [:, :, self.offset:self.offset + T] = k
        self.values[:, :, self.offset:self.offset + T] = v
        self.offset += T
        return self.keys[:, :, :self.offset], self.values[:, :, :self.offset]

    def reset(self):
        self.offset = 0
        self.keys.zero_()
        self.values.zero_()
```

At seq_len=1024: naive costs 1,048,576 ops; KV cache costs ~192 — a **~5000× speedup**.

### 8.2 Temperature Sampling

```python
def apply_temperature(logits, temperature=1.0, cap=30.0):
    capped = cap * torch.tanh(logits / cap)
    probs  = torch.softmax(capped / temperature, dim=-1)
    return probs
```

| Temperature | Effect |
|---|---|
| T → 0 (greedy) | Always picks the most likely token; deterministic |
| T = 0.2 | Very conservative, near-deterministic |
| T = 1.0 | Balanced (recommended default) |
| T > 1.5 | Creative / unpredictable |

### 8.3 Top-p (Nucleus) Sampling

Only sample from the smallest set of tokens whose cumulative probability ≥ `top_p`:

```python
def top_p_sample(logits, temperature=1.0, top_p=0.9, cap=30.0):
    probs = torch.softmax(cap * torch.tanh(logits / cap) / temperature, dim=-1)
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_probs   = probs[sorted_indices].cumsum(dim=-1)
    mask = sorted_probs <= top_p
    mask[sorted_indices[0]] = True       # always keep at least one token
    filtered = probs * mask.float()
    filtered /= filtered.sum()           # renormalise
    return torch.multinomial(filtered, 1).item()
```

| Method | When to use |
|---|---|
| Greedy (T=0 / argmax) | Evaluation, reproducibility |
| Temperature only | Simple diversity control |
| Top-p = 0.9, T = 0.5–1.0 | Production generation |

---

## Part 9 — Training Loop (Notebook 10)

### 9.1 Loss Function

Language modelling uses **cross-entropy** on next-token prediction:

```python
# logits: (B, T, vocab_size)  target: (B, T)
# Target at position i is the NEXT token (shift by 1)
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, vocab_size),   # predictions for positions 0..T-2
    target_ids[:, 1:].reshape(-1)                # targets are positions 1..T-1
)
```

### 9.2 One Training Step

```python
def train_single_step(model, input_ids, target_ids, optimizer, loss_fn):
    optimizer.zero_grad()

    logits = model(input_ids)                        # forward pass
    loss = loss_fn(
        logits[:, :-1, :].reshape(-1, vocab_size),
        target_ids[:, 1:].reshape(-1)
    )

    loss.backward()                                  # backprop
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip
    optimizer.step()                                 # update

    return loss.item()
```

**Gradient clipping** (`max_norm=1.0`) is essential — it prevents occasional very large gradients from destabilising training.

### 9.3 Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

AdamW adds **weight decay** (L2 regularisation) decoupled from the adaptive gradient. Preferred over Adam for transformer training.

### 9.4 Full Training Loop

```python
model.train()
for step in range(n_steps):
    input_ids, target_ids = dataset[step % len(dataset)]
    loss = train_single_step(model, input_ids.unsqueeze(0),
                              target_ids.unsqueeze(0), optimizer, loss_fn)
```

**What real Gemma 3 training adds beyond this:**
- 2–14 trillion tokens of training data
- TPU v5p/v5e clusters with ZeRO-3 sharding
- Knowledge distillation from a larger teacher model
- Post-training RLHF / BOND / WARM alignment
- Multi-epoch cosine LR schedule with warmup

---

## Part 10 — Key PyTorch Commands Reference

### Tensor Operations

```python
# Shapes and views
x.shape                              # torch.Size([B, T, D])
x.view(B, T, n_heads, head_dim)     # reshape (contiguous required)
x.transpose(1, 2)                    # swap dims 1 and 2
x.contiguous()                       # make memory contiguous after transpose
x.unsqueeze(0)                       # add dimension at position 0
x.squeeze(-1)                        # remove last dim if size 1

# Math
torch.matmul(A, B)                   # batched matrix multiply
torch.outer(a, b)                    # outer product (1D tensors)
torch.rsqrt(x)                       # 1 / sqrt(x), element-wise
x.pow(2).mean(-1, keepdim=True)      # RMS computation
x.cumsum(dim=-1)                     # cumulative sum (used in top-p)
x.repeat_interleave(n, dim=1)        # repeat elements (used in GQA)

# Masking
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores.masked_fill(mask, float('-inf'))   # set masked positions to -inf

# Normalisation
F.normalize(x, p=2, dim=-1)          # L2 normalisation (QK-Norm)
F.softmax(x, dim=-1)                 # softmax over last dim
F.gelu(x, approximate='tanh')        # GELU with tanh approximation (GeGLU)
F.silu(x)                            # SiLU = x × sigmoid(x) (SwiGLU)
```

### Module Patterns

```python
# Linear projection (no bias — Gemma style)
nn.Linear(d_in, d_out, bias=False)

# Embedding
nn.Embedding(vocab_size, hidden_size)    # lookup table

# Parameter
nn.Parameter(torch.zeros(dim))           # learnable tensor, registered as param

# ModuleList
nn.ModuleList([block for block in blocks])   # list of modules; properly registered

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Training Utilities

```python
model.train()                            # enable dropout/batchnorm training mode
model.eval()                             # disable them
torch.no_grad()                          # context manager: no gradient tracking
optimizer.zero_grad()                    # clear accumulated gradients
loss.backward()                          # compute gradients
optimizer.step()                         # apply gradients

# Count parameters
sum(p.numel() for p in model.parameters())
```

### Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    try:
        torch.zeros(1).to(device)
    except Exception:
        device = torch.device('cpu')
model = model.to(device)
tensor = tensor.to(device)
tensor.cpu()                             # move back to CPU
```

---

## Part 11 — Notebook Suitability Assessment

### Overall Verdict: ✅ Well-structured and presentation-ready

The notebooks are progressive, hands-on, and focused. Each introduces one component, exercises it in isolation, then feeds into the next notebook. The 11-notebook arc builds to a complete working model — a compelling workshop narrative.

### Strengths

- Every architectural innovation is contrasted with its predecessor (e.g. QK-Norm vs soft-capping, GeGLU vs SwiGLU, Double-norm vs Pre-norm).
- Exercises have solutions embedded (collapsed or in comments).
- Estimated times are realistic (~10–20 min per notebook).
- The toy dataset and tokenizer keep everything self-contained without requiring downloads.

### Issues to Address Before PyCon

| Notebook | Issue | Fix |
|---|---|---|
| **09** | `import matplotlib.pyplot as plt` is missing but `plt` is called | Add import |
| **09** | `top_p_sampling` has a bug: `mask[sorted_indices[0]] = True` operates on wrong indexing | Rewrite mask application |
| **05** | Docstring mentions "Gemma 3 used SwiGLU" and "Gemma 3 found GELU" — confusing since both are called Gemma 3 | Clarify as Gemma 2 vs Gemma 3 |
| **04** | `apply_rope` splits into left/right halves but the standard interleaved rotation is slightly different — worth a comment | Add a note on the convention |
| **10** | Hard-coded path `sys.path.insert(0, '/home/lmassaron/code/...')` | Remove or replace with relative path |
| **All** | `SimpleTokenizer` is redefined in notebooks 05, 08, 09, 10 with slight differences | Extract to a shared `utils.py` |

### Suggested Additions

- A **cheat sheet slide / one-pager** summarising the full architecture (this document covers it).
- Notebook 08 could include `model.load_state_dict()` showing how to load official Gemma 3 weights — fulfilling the "loads official weights" promise in the abstract.
- Add timing comparisons between naive attention and KV cache with actual wall-clock numbers.

---

## Summary: Gemma 3 vs Gemma 2 Differences

| Component | Gemma 2 | Gemma 3 |
|---|---|---|
| Attention score stabilisation | `tanh(scores / 50.0)` | **QK-Norm** (L2 normalise Q, K) |
| FFN gate activation | SwiGLU (`SiLU`) | **GeGLU** (`GELU`) |
| Context length | 8K | **128K** (via θ=8M RoPE in global layers) |
| Final logit cap | 30.0 | 30.0 (unchanged) |
| Normalisation | Pre-norm only | **Double-norm** (Pre + Post) |
| KV heads | 2 (GQA) | 2 (GQA, unchanged) |
| Local/global pattern | 5:1 | 5:1 (unchanged) |
