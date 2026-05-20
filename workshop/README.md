# Building Gemma 3 from Scratch

A hands-on 2-hour workshop that takes you from attention math to a fully working Gemma 3 model, implementation by implementation.

## What is Gemma 3?

Gemma 3 is Google's open-weight multilingual, multimodal language model family. In this workshop we focus on the **text-only architecture** and replicate it at scale, building a ~270M parameter model (a compact subset of the 1B–27B range).

### Key Gemma 3 Innovations

| Feature | Description |
|---|---|
| **QK-Norm** | Query-key L2-normalization replaces the logit soft-capping used in Gemma 2 |
| **GQA** | Grouped-Query Attention: shared KV heads reduce KV-cache memory |
| **Sliding Window** | 1024-token local attention + global layers interleaved 5:1 |
| **Dual RoPE** | 10K frequency for local layers, 1M for global layers |
| **GeGLU** | Gaussian Gated Linear Unit (not SwiGLU) as the FFN activation |
| **Double RMSNorm** | Pre-norm AND post-norm for every sub-layer (unique to Gemma 2/3) |
| **Logit Soft-Capping** | Final output logits capped at 30.0 via tanh for training stability |
| **262K Vocabulary** | Byte-level BPE tokenizer (same base as Gemini 2.0) |
| **128K Context** | Via RoPE base-frequency rescaling (×8) |

## Table of Contents

| # | Notebook | Topic | Time |
|---|---|---|---|
| 0 | [Setup & Tokenization](00_setup_and_tokenization.ipynb) | Vocabulary, BPE, embedding layer | 10 min |
| 1 | [The Math of Attention](01_the_math_of_attention.ipynb) | QK-Norman + Scaled Dot-Product Attention | 15 min |
| 2 | [Causal Masking](02_causal_masking.ipynb) | Global + sliding window masks | 10 min |
| 3 | [Grouped Query Attention](03_grouped_query_attention.ipynb) | GQA: shared KV heads | 15 min |
| 4 | [RoPE](04_rotary_positional_embeddings.ipynb) | Positional embeddings with dual frequencies | 15 min |
| 5 | [GeGLU](05_gated_mlp.ipynb) | Gated MLP activation (Gemmma 3 uses GeGLU) | 10 min |
| 6 | [RMSNorm](06_rmsnorm_and_normalization.ipynb) | Gemma-style double normalization | 10 min |
| 7 | [Transformer Block](07_the_transformer_block.ipynb) | Putting it all together in one layer | 20 min |
| 8 | [Model Assembly](08_gemma_model_assembly.ipynb) | Full Gemma-3Model + text generation | 15 min |
| 9 | [Inference](09_inference_and_sampling.ipynb) | KV cache, temperature, top-p / top-k | 10 min |
| 10 | [Training Loop](10_training_loop.ipynb) | Mini training on toy data | 10 min |
| | **BUFFER / Q&A** | | **10 min** |
| | **Total** | | **~130 min** |

## Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0
- `matplotlib` for visualizations
- Basic familiarity with linear algebra (matrices, dot products) and deep learning (tensors, gradients)

## Setup

```bash
pip install -r requirements.txt
```

Ensure a GPU is available for faster execution (CPU also works for all notebooks).

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

## Architecture Summary: Gemma 3 ~270M Target

```
┌─────────────────────┬──────────────┐
│ vocab_size          │   256,000    │  (simplified from 262K)
│ hidden_size         │      768     │
│ num_layers          │        8     │
│ num_attention_heads │        8     │
│ num_kv_heads        │        2     │
│ intermediate_size   │     2048     │
│ sliding_window      │     1024     │
│ max_position_embed  │      512     │
│ context_length      │      512     │
│ QK-Norm             │      True    │
│ logit_cap           │      30.0    │
└─────────────────────┴──────────────┘
```

## Architecture Diagram

```
Input IDs
    │
    ▼
┌──────────┐    ┌──────────────────────────────────────┐
│ Embedding│───►│  Transformer Blocks (×8)              │
│ (+scale) │    │  ┌──────────────────────────────────┐ │
└──────────┘    │  │  Local (sw=1024, f=10K)  ◄───────┤ │
                │  │  Global    (sw=∞,   f=1M)   ◄─────┤ │
                │  │  5 local + 1 global pattern          │ │
                │  └──────────────────────────────────────┘ │
                └──────────────────────────────────────────┘
    │
    ▼
┌──────────┐
│ RMSNorm  │
└────┬─────┘
     │
     ▼
┌──────────┐     ┌───────────┐
│ LM Head  │────►│ Logit Cap │──► Next Token(s)
│ (weight  │     │ =30.0     │     (tanh(logits/30))
│  tied)   │     └───────────┘
└──────────┘
```

## Gemma 2 vs Gemma 3 Comparison

| Component | Gemma 2 | Gemma 3 |
|---|---|---|
| Attention | Soft-capping (tanh at 50) | **QK-Norm** (L2 normalize Q & K) |
| MLP | SwiGLU | **GeGLU** |
| KV Cache | 1:1 local:global ratio | **5:1** local:global (smaller KV cache) |
| Sliding Window | 4096 | **1024** |
| RoPE base | 10K (all layers) | **10K local + 1M global** |
| Context | 32K → 8K | **128K** |
| Final logit cap | 30.0 | **30.0** (preserved) |
| RMSNorm init | weight=1 | **weight=0, scale=(1+weight)** |
| Normalization | Pre + Post (attention) | **Pre + Post (both attention + MLP)** |

## License

This workshop materials are released under MIT License. Gemma models are under the Gemma license — see https://github.com/google/gemma_models

## Acknowledgements

Built with reference to the [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) by the Gemma Team at Google.
