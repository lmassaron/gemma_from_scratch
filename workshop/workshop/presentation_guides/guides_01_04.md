# Presentation Guide: Notebook 1 - The Math of Attention

## 1. Introduction (2 mins)
*   **The Goal:** We are building the "eye" of the model. Attention is how a model decides which words in a sentence are relevant to each other.
*   **The Hook:** In traditional programming, we use `if/else`. In LLMs, we use "soft" lookups called Attention.

## 2. Cell-by-Cell Walkthrough

### Cell 3: The Library Analogy (Crucial!)
*   **The Analogy:** 
    *   **Query:** What you typed in Google.
    *   **Key:** The titles of the search results.
    *   **Value:** The content of the pages.
*   **Talking Point:** "We compare the Query to every Key to see how well they match."

### Cell 8: Scaling & Soft-Capping (Advanced Insight)
*   **Scaling:** We divide by `sqrt(dk)` to keep values small.
*   **Soft-Capping:** This is a Gemma 2/3 innovation. We use `tanh` to "squash" the scores so they never exceed 50.0.
*   **Talking Point:** "Soft-capping is a safety mechanism. It prevents the model from becoming too 'confident' or 'stubborn' during training, which makes it much more stable."

### Cell 10: Softmax & Visualization
*   **Visual Aid:** Look at the heatmap. Bright spots = "This word is paying attention to that word." 
*   **Property:** Each row sums to 1.0.

## 3. The Exercise (5 mins)
*   **Task:** Implement `scaled_dot_product_attention` with the `tanh` cap.
*   **Discussion:** Ask: "What happens if a logit reaches 1000?" (Answer: Without soft-capping, `exp(1000)` would overflow. Soft-capping keeps it at 50.0).

---
# Presentation Guide: Notebook 2 - Causal Masking

## 1. Introduction (1 min)
*   **The Concept:** Prevent the model from "cheating" by looking at future words during training.
*   **Pro-Tip:** "Gemma 2 alternates between Global and Local (Sliding Window) masks to get the best of both worlds."

---
# Presentation Guide: Notebook 3 - Grouped Query Attention

## 1. Introduction (2 mins)
*   **Analogy:** A classroom where 4 students (Queries) share 1 textbook (Key/Value). It saves space!
*   **Talking Point:** "Gemma 2 (9B/27B) uses exactly 2 KV groups. This is a highly optimized ratio."

---
# Presentation Guide: Notebook 4 - RoPE

## 1. Introduction (3 mins)
*   **Analogy:** Instead of adding a number to the word, we *rotate* the word's vector. 
*   **Talking Point:** "RoPE is the industry standard now. It's used in Llama, Gemma, and Mistral because it handles long sequences beautifully."
