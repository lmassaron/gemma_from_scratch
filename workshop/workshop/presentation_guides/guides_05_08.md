# Presentation Guide: Notebook 5 - Gated MLP (GeGLU)

## 1. Introduction (2 mins)
*   **Talking Point:** "Gemma uses GeGLU. The 'Ge' stands for GELU. It's more sophisticated than the old SwiGLU used in Llama."

---
# Presentation Guide: Notebook 6 - RMSNorm

## 1. Introduction (2 mins)
*   **The "Double-Norm" Secret:** "Gemma 2/3 is unique because it normalizes *both* the input and the output of every sub-layer. This is why it's so incredibly stable even when it gets very deep."

---
# Presentation Guide: Notebook 7 - The Transformer Block

## 1. Cell-by-Cell Walkthrough
### Cell 2: Architecture
*   **Discussion:** Point out the `input_layernorm` (Pre) and `post_attention_layernorm` (Post). 
*   **Advanced Tip:** "If you were building Gemma 2, you would alternate the mask between 'Local' (Sliding Window) and 'Global' every other layer. This allows the model to have 'local focus' and 'global context' at the same time."

---
# Presentation Guide: Notebook 8 - Model Assembly

## 1. Cell-by-Cell Walkthrough

### Cell 3: Final Soft-Capping (30.0)
*   **Explanation:** "Just like we capped the attention scores at 50.0, we cap the *final* outputs at 30.0."
*   **Why 30.0?** "It was found through experimentation that 30.0 is the 'sweet spot' for keeping the final probabilities from becoming too extreme."

### Cell 3: Autoregressive Generation
*   **Safety Tip:** Reiterate the need to crop the sequence to `context_length`. "The model's RoPE embeddings and Masks are designed for a specific length. If you exceed it, the model becomes 'confused'."

## 3. Final Demo (5 mins)
*   **Closing:** "You've just built a mini-Gemma. From individual dot products to a full text-generation engine, you've implemented the key innovations that make Google's open models world-class."
