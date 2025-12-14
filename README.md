# Gemma From Scratch

This repository provides a clear and minimal implementation for training a Gemma-like language model from scratch using PyTorch. The project is structured to be easily understandable, with a clear separation between the core model logic and the training/data preparation scripts.

The implementation is heavily inspired by and builds upon the foundational work from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Two Workflows: Training vs. Inference

This repository supports two primary use cases:
  
1.  Inference with Official Gemma Weights: The inference_google_gemma.py script uses the official Gemma tokenizer and loads the pre-trained 270M model from Hugging Face. This demonstrates the architectural compatibility of our implementation.
2.  Training a Model From Scratch: The prepare_dataset.py and train.py scripts allow you to train a model on your own data (e.g., TinyStories). For this workflow, we use the simpler and faster GPT-2 tokenizer (tiktoken). This is a self-contained training pipeline, and the resulting model should be used with inference_custom.py, which also uses the GPT-2 tokenizer.

## Model Architecture

The Gemma 3 model is a decoder-only transformer, similar to other popular language models. It is composed of the following key components:

*   **Token Embeddings:** The model starts by converting input tokens into dense vectors of a specified dimension (`emb_dim`).
*   **Transformer Blocks:** The core of the model is a series of transformer blocks (`n_layers`), each containing:
    *   **Attention Mechanism:** Gemma 3 uses a sophisticated attention mechanism that combines global and sliding-window attention. This allows the model to handle long contexts efficiently while still capturing global dependencies.
    *   **Feed-Forward Network:** A position-wise feed-forward network with a SwiGLU activation function.
    *   **Layer Normalization:** RMSNorm is used for layer normalization.
*   **Rotary Positional Embeddings (RoPE):** RoPE is used to inject positional information into the model.
*   **Output Head:** A final linear layer that projects the output of the transformer blocks back to the vocabulary space to produce logits.

This implementation supports both a custom configuration and the Gemma 3 270M configuration, as defined in the `config.py` file.

## How it Works

The model takes a sequence of tokens as input and processes them through the embedding layer. The resulting embeddings are then passed through a series of transformer blocks. Each block applies self-attention and a feed-forward network to the input. The attention mechanism is masked to prevent tokens from attending to future tokens, and it can be either global or sliding-window, depending on the layer's configuration.

The final output of the transformer blocks is then passed through a normalization layer and a linear layer to produce the logits for the next token in the sequence.

During inference, the model generates text autoregressively. It takes a starting sequence of tokens, predicts the next token, appends it to the sequence, and repeats the process until a specified number of new tokens have been generated.

## Project Structure

For better organization and modularity, the project files have been arranged as follows. This separates the user-facing scripts from the core, importable Python package.

```
gemma_from_scratch/
├── README.md
├── requirements.txt
│
├── prepare_dataset.py           # User-facing script to download and process data
├── train.py                     # User-facing script to train the model
├── inference_custom.py          # Test the inference capabilities of your custom model
├── inference_google_gemma.py    # Test the inference capabilities of the original Gemma 270M
│
├── gemma_scratch/            # Core source code as a Python package
│   ├── __init__.py         # Makes 'gemma_scratch' a package
│   ├── config.py           # Model hyperparameters
│   ├── layers.py           # The TransformerBlock implementation
│   ├── model.py            # The Gemma model definition
│   ├── normalization.py    # RMSNorm implementation
│   ├── rope.py             # RoPE (Rotary Positional Embeddings) implementation
│   └── tokenizer.py        # Tokenizer utilities
├── evals/                    # Scripts and utilities for model evaluation
│   ├── EVAL.md             # Documentation for evaluation
│   └── evaluate_model.py   # Script to evaluate the trained model
└── utils/                    # Utility scripts for various tasks
    ├── count_parameters.py       # Script to count model parameters
    ├── count_tokens.py           # Script to count tokens in a dataset
    ├── generate_model_summary.py # Script to generate a model summary
    ├── GPU_performance_benchmark.py # Script for GPU performance benchmarking
    └── model_summary.txt         # Placeholder for model summary output
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lmassaron/gemma_from_scratch.git
    cd gemma_from_scratch
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## How to Use

The process is divided into two main steps: data preparation and model training.

### Step 1: Prepare the Dataset

This script downloads a dataset from the Hugging Face Hub, tokenizes it using a GPT-2 tokenizer, and saves the token sequences into efficient binary files (`.bin`) for rapid loading during training.

*   **To run with the default dataset (`roneneldan/TinyStories`):**
    ```bash
    python prepare_dataset.py
    ```
    This command will create a directory named `tinystories_data/` containing `train.bin` and `validation.bin`.

*   **To use a different dataset from the Hub and a custom output directory:**
    ```bash
    python prepare_dataset.py <huggingface_dataset_name> --output-dir ./data
    ```
    For example: `python prepare_dataset.py "c4" --data-subset "en.noblocklist" --output-dir ./c4_data`

### Step 2: Train the Model

Once the dataset is prepared, you can start training the model. This script handles the entire training loop, including optimization, learning rate scheduling, and periodic evaluation.

*   **To train using the default dataset location:**
    *(This assumes you ran the default `prepare_dataset.py` command in the previous step)*
    ```bash
    python train.py
    ```

*   **To point the training script to a custom data directory:**
    *(This is required if you used the `--output-dir` option when preparing the data)*
    ```bash
    python train.py --data-dir ./data
    ```

The training script will save the following outputs in the root directory:
*   `best_model_params.pt`: The state dictionary of the model that achieved the lowest validation loss.
*   `loss_plot.png`: A plot showing the training and validation loss curves over time.

### Key Components & Logic

*   **`prepare_dataset.py`**: A flexible data processing script. It parallelizes the tokenization step across all available CPU cores for maximum efficiency and uses memory-mapped NumPy arrays to handle datasets larger than RAM.

*   **`train.py`**: The main training loop. It implements modern training best practices:
    *   **Mixed-Precision Training:** Uses `torch.amp.autocast` with `bfloat16` for faster training and reduced memory usage on supported hardware.
    *   **Optimizer:** Employs the AdamW optimizer, which adds weight decay for better regularization.
    *   **Learning Rate Scheduler:** Uses a `SequentialLR` scheduler that combines a linear warmup phase with a cosine decay, helping to stabilize training.
    *   **Gradient Accumulation:** Allows for training with large effective batch sizes even on memory-constrained GPUs.
    *   **Gradient Clipping:** Prevents exploding gradients by clipping the norm of the gradients before the optimizer step.

*   **`gemma_scratch/` (The Core Package):**
    *   `model.py`: Defines the `Gemma3Model` class, a PyTorch `nn.Module` that assembles the complete transformer architecture.
    *   `layers.py`: Contains the `TransformerBlock`, the core repeating unit of the model, which includes multi-head attention and the MLP layers.
    *   `rope.py`: Implements **Rotary Positional Embeddings (RoPE)**, a modern technique for injecting positional information into the self-attention mechanism.
    *   `normalization.py`: Provides an efficient `RMSNorm` (Root Mean Square Normalization) layer, which is used throughout the Gemma architecture instead of traditional LayerNorm.
    *   `config.py`: A simple file to store the model's hyperparameters (e.g., number of layers, heads, embedding dimensions).
    *   `tokenizer.py`: A wrapper for the GPT-2 tokenizer used for encoding the text data.

## Usage

This repository provides two main scripts for running inference:

*   **`inference_google_gemma.py`**: This script uses the pre-trained Gemma 3 270M model from the Hugging Face Hub. It downloads the weights, loads them into the model, and generates text from a list of predefined sentences. This is the recommended script for most users.

    ```bash
    python inference_google_gemma.py
    ```

*   **`inference_custom.py`**: This script is for running inference with a custom model. It requires a `.pth` file with the model weights. You can use this script to test your own trained models.

    ```bash
    python inference_custom.py --model-path /path/to/your/model.pth
    ```

Both scripts will output the generated text to the console.

## References

*   **Original Notebook:** [LLMs-from-scratch/ch05/12_gemma3/standalone-gemma3.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/standalone-gemma3.ipynb) by Sebastian Raschka.
*   **Hugging Face Model Card:** [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.
