# Gemma 3 from Scratch

This repository contains a from-scratch implementation of the Gemma 3 model, a family of lightweight, state-of-the-art open models from Google. This project is based on the original notebook by Sebastian Raschka, which can be found [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/standalone-gemma3.ipynb).

## About this Project

This project aims to provide a clear and understandable implementation of the Gemma 3 model, making it accessible for educational and research purposes. The code is structured into several files, each with a specific purpose, to facilitate a modular and organized understanding of the model's architecture and functionality.

## Getting Started

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/gemma-3-from-scratch.git
    cd gemma-3-from-scratch
    ```

2.  **Create a virtual environment and install the dependencies:**

    This project uses `uv` to manage the virtual environment and dependencies.

    ```bash
    bash install.sh
    ```

    To activate the virtual environment, run:

    ```bash
    source .venv/bin/activate
    ```

### Running the Inference

To run the inference with the pre-trained Gemma 3 270M model, you can use the `inference_google_gemma.py` script:

```bash
python inference_google_gemma.py
```

This script will download the model weights from the Hugging Face Hub, load them into the model, and generate text based on a set of predefined sentences.

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

## Usage

This repository provides two main scripts for running inference:

*   **`inference_google_gemma.py`**: This script uses the pre-trained Gemma 3 270M model from the Hugging Face Hub. It downloads the weights, loads them into the model, and generates text from a list of predefined sentences. This is the recommended script for most users.

    ```bash
    python inference_google_gemma.py
    ```

*   **`inference_custom.py`**: This script is for running inference with a custom model. It requires a `.pth` file with the model weights. You can use this script to test your own trained models.

    ```bash
    python inference_custom.py --pth_path /path/to/your/model.pth
    ```

Both scripts will output the generated text to the console.

## Files

*   **`config.py`**: Contains the configurations for the Gemma 3 model, including `GEMMA3_CONFIG_CUSTOM` and `GEMMA3_CONFIG_270M`.
*   **`model.py`**: Defines the Gemma 3 model architecture, including the `Gemma3Model` class and the function to load pre-trained weights.
*   **`layers.py`**: Contains the building blocks of the transformer model, such as the `TransformerBlock` and `Attention` classes.
*   **`normalization.py`**: Implements the `RMSNorm` layer.
*   **`rope.py`**: Contains the functions for computing the Rotary Positional Embeddings (RoPE).
*   **`tokenizer.py`**: A simple tokenizer for the Gemma 3 model.
*   **`inference_google_gemma.py`**: A script for running inference with the pre-trained Gemma 3 270M model from the Hugging Face Hub.
*   **`inference_custom.py`**: A script for running inference with a custom model.
*   **`train_from_scratch.py`**: A script for training the Gemma 3 model from scratch (this is a placeholder and not fully implemented).
*   **`install.sh`**: A bash script to create a virtual environment and install the required dependencies.

## References

*   **Original Notebook:** [LLMs-from-scratch/ch05/12_gemma3/standalone-gemma3.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/standalone-gemma3.ipynb) by Sebastian Raschka.
*   **Hugging Face Model Card:** [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.
