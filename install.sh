#!/bin/bash

# --- Configuration ---
VENV_NAME=".venv"
PYTHON_VERSION="3.12"

# --- Common packages for both platforms ---
# Note: torch is handled separately below
COMMON_PACKAGES=(
    'datasets'
    'tiktoken'
    'tokenizers'
    'safetensors'
    'numpy'
    'tqdm'
    'protobuf'
    'tensorboard'
    'matplotlib'
)

# --- Environment Setup ---
echo "Creating virtual environment..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --seed

# --- OS-Specific PyTorch Installation ---
# Detect the operating system (macOS is 'Darwin')
if [ "$(uname -s)" = "Darwin" ]; then
    # --- macOS (Apple Silicon / Intel) Installation ---
    echo "Detected macOS. Installing PyTorch with MPS support..."
    uv pip install -U \
        --python "$VENV_NAME/bin/python" \
        'torch' \
        "${COMMON_PACKAGES[@]}"

else
    # --- Linux (or other OS) Installation with CUDA ---
    echo "Detected non-macOS system. Installing PyTorch with CUDA support..."
    uv pip install -U \
        --python "$VENV_NAME/bin/python" \
        'torch' --extra-index-url https://download.pytorch.org/whl/cu121 \
        "${COMMON_PACKAGES[@]}"
fi

echo ""
echo "✅ Installation complete."
echo "To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"