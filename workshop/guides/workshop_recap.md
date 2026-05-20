# Workshop Recap: Building Gemma 3 From Scratch

---

## 🧱 The Architecture at a Glance

### Gemma 3 (Pedagogical Scale: ~270M Parameters)

```
                              ┌──────────────────────────────────────────────┐
                              │           GEMMA 3 MODEL                      │
                              └──────────────────────────────────────────────┘

  INPUT TOKENS                                              OUTPUT PROBS
  ┌──────────┐                                                 ┌──────────┐
  │  Text    │  ┌──────────┐  ┌────────────────┐  ┌─────────┐│  ┌──────────┐  │
  │ "Hello   │→│ Embedding│→│  8 Transformer  │→│ RMSNorm ││→│ softmax  │→│ Next Token│
  │ World"   │  │ × √768   │  │  blocks (5+1)  │  │         ││  └──────────┘  │
  └──────────┘  └──────────┘  └────────────────┘  └─────────┘│  │  logit cap  ││
                              └──────────────────────────────────────────────┘
                              │  tanh(x/30), logit cap                   │
                              │  lm_head weight-tied to embeddings         │
                              └──────────────────────────────────────────────┘
```

---

## 🧩 Notebook-by-Notebook Architecture Map

| Notebook | Topic | Key Gemma 3 Innovation | Tensor Shapes |
|----------|-------|----------------------┬---------------|---------||
| **00** | Tokenization & Embedding | 262K BPE vocab | `[batch, seq_len] → [batch, seq_len, 768]` |
| **01** | QK-Norm Attention | QK-L2 normalization → cosine similarity | `[batch, 8, seq, 96]` |
| **02** | Causal Masking | 5:1 local/global pattern | `[seq, seq]` boolean mask |
| **03** | Grouped Query Attention | 8Q:2KV GQA → 4× KV-cache savings | `[batch, 2, seq, 96]` → expand to 8 |
| **04** | Rotary Positional Embeddings | Dual frequency: θ=10K (local) / θ=8M (global) | `sin/cos` rotation per head_dim pair |
| **05** | Gated MLP (GeGLU) | 3-projection GELU gating, intermediate_size=2048 | `[batch, 768] → [batch, 2048] → ...` |
| **06** | RMSNorm (Double-Norm) | 4 calls per block, weight=0 init, (1+weight) scaling | per-sublayer normalization |
| **07** | Transformer Block Assembly | Combines all previous components in one block | See diagram above |
| **08** | Full Model Assembly | Weight tying, embedding scaling, logit cap | → `tanh(x/30)` → softmax → next token |
| **09** | Inference & Sampling | KV cache, temperature, top-p nucleus sampling | K,V cache: `[batch, t, 2, 96]` |
| **10** | Training Loop | Cross-entropy, AdamW, gradient clipping, cosine LR | loss from `random ≈ 12.5` → converges |

---

## 🔑 Gemma 3 vs. Other Models (Quick Reference)

| Feature | Gemma 3 | Llama 3 | GPT-4 (inferred) |
|---------|--------|--------|-----------------|
| **Attention** | GQA (8:2) | GQA | MHA or GQA |
| **Position Encoding** | RoPE (dual freq: 10K/8M) | RoPE (single freq: 500K) | RoPE (unknown) |
| **MLP** | GeGLU (GELU gating) | SwiGLU (SiLU gating) | SwiGLU |
| **Normalization** | Double RMSNorm (pre + post) | Single RMSNorm (pre only) | RMSNorm variants |
| **Attention Score** | QK-Norm | QK-Norm | QK-Norm |
| **Logit Cap** | tanh(x/30) | tanh(x/30) | unknown |
| **Layer Pattern** | 5 local + 1 global | Standard | Standard |
| **Weight Tying** | Yes | No | No |

---

## 📐 Parameter Budget (Pedagogical Model)

| Component | Formula | Parameters |
|-----------|--------|-----------|
| Embeddings | vocab × hidden_size | 262K × 768 ≈ **200M** |
| Attention proj | 4 × hidden_size² | 4 × 768² ≈ **2.3M** |
| MLP gate/up/down | 3 × hidden_size × intermediate_size | ~**4.4M** |
| RMSNorm (4 × block) | 4 × 768 × 8 | ~**24K** |
| **Total** | | **~270M** (mostly embeddings) |

*Note: Embeddings dominate the parameter budget (~74%).*

---

## 🎯 Teaching Timeline (2h Workshop)

| Block | Time | Content |
|-------:|------|-------|
| **Warm-up** | 5 min | Icebreakers, setup check, what we'll build |
| **Block 1** | 40 min | Notebooks 00–04 (vision: tokenization → attention → RoPE) |
| *Break* | 5 min | Stretch, Q&A, refresh |
| **Block 2** | 70 min | Notebooks 05–08 (thinking: MLP → normalization → block → assembly) |
| *Break* | 5 min | Stretch, Q&A, refresh |
| **Block 3** | 30 min | Notebooks 09–10 (inference & training) |
| **Close** | 5 min | Recap diagram, resources, next steps |

---

## 🚦 Teaching Tips by Notebook

### Which Notebooks Students Struggle With
| Notebook | Difficulty | Common Blockers |
|----------|-----------|----------------|
| 00 | Easy | None |
| 01 | Easy | Confusing "cosine similarity" |
| 02 | Medium | Understanding sliding window |
| 03 | Medium | When/why to use `repeat_interleave` |
| 04 | **Hard** | RoPE rotation math |
| 05 | Medium | When/why gating matters |
| 06 | Medium | Double-norm confusion |
| 07 | **Hard** | Putting it all together |
| 08 | Easy | Just running the code |
| 09 | Medium | KV cache concept is dense |
| 10 | Medium | Understanding loss curves |

### When to Rush vs. Slow Down
- **Rush:** Notebooks 00, 01 (intuitive), 08 (just runs the code)
- **Slow down:** Notebooks 04 (RoPE math), 07 (assembly), 09 (KV cache)
- **Let students explore:** Notebook 05 (gating exercise), Notebook 06 (RMSNorm weight=0), Notebook 09 (temperature/top-p demos)

### When to Show the Solution to the Exercise
- **Show immediately:** None in blocks 1–2 (let them type it)
- **Show if stuck > 5 min:** Notebook 03 GQA + Notebook 5 GeGLU exercise
- **Show only on request:** Notebook 04 RoPE (students should feel the "aha" moment)

---

## ⚠️ Common Presenter Mistakes (Avoid These!)

1. **Don't skip the weight tying explanation.** Students expect the output layer to be a separate matrix. Explain *why* tying is used (regularization + parameter savings + learned duality between input and output space).
2. **Don't say "RMSNorm is just LayerNorm without mean."** You must emphasize that RMSNorm is *not* LayerNorm with one subtraction removed. It has different gradient properties and different initialization requirements.
3. **Don't rush the KV cache demo.** The speed difference (compute vs. cache) is the most memorable moment of the workshop. Show it with actual timing.
4. **Don't present SwiGLU vs GeGLU as "Gemma 3 chose GeGLU because X".** Be honest: both work. Gemma 3 uses GeGLU; Llama uses SwiGLU. The architectural choice is minor compared to data quality and scale.
5. **Don't skip the loss curve visualization.** Students need to see what training progression *should* look like. Show the curve: steep drop → plateau → gradual improvement.

---

## 📚 Additional Resources to Share with Students

1. **[Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)** — The official paper
2. **[RoPE Paper (Su et al. 2021)](https://arxiv.org/abs/2104.09864)** — Rotary positional embeddings
3. **[GQA Paper (Ding et al. 2023)](https://arxiv.org/abs/2305.13245)** — Grouped Query Attention
4. **[Shazeer 2020 (GeGLU)](https://arxiv.org/abs/2002.05202)** — Gated Linear Units
5. **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — Production implementation

---

## 🏁 What Students Have Built by End of Workshop

```
✅ A complete 270M-parameter Gemma 3 decoder
✅ Trained on a small toy dataset
✅ Generates text with KV cache acceleration
✅ Samples with temperature and top-p
✅ Understands: QK-Norm, GQA, RoPE, GeGLU, double RMSNorm, weight tying
✅ Can explain how every Gemma 3 innovation works mathematically
```
