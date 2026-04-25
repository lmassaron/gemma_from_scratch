#!/bin/bash

# --- Configuration ---
VENV_NAME=".venv"
PYTHON_VERSION="3.12"

# --- Environment Setup ---
echo "Creating virtual environment..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Using python -m venv..."
    python3 -m venv "$VENV_NAME"
    PIP_CMD="$VENV_NAME/bin/python -m pip"
    INSTALL_ARGS=""
else
    uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --seed
    PIP_CMD="uv pip"
    INSTALL_ARGS="--python $VENV_NAME/bin/python"
fi

# --- OS-Specific PyTorch Installation ---
# Detect the operating system (macOS is 'Darwin')
if [ "$(uname -s)" = "Darwin" ]; then
    # --- macOS (Apple Silicon / Intel) Installation ---
    echo "Detected macOS. Installing PyTorch with MPS support..."
    $PIP_CMD install -U \
        $INSTALL_ARGS \
        'torch'

else
    # --- Linux (or other OS) Installation with CUDA detection ---
    echo "Detected non-macOS system. Checking for CUDA..."
    
    CUDA_VERSION=""
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n 1)
    elif command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -n 1)
    fi

    if [ -n "$CUDA_VERSION" ]; then
        echo "Detected CUDA version: $CUDA_VERSION"
        
        # Map CUDA version to PyTorch wheel index
        # PyTorch currently supports cu118, cu121, cu124
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            CUDA_TAG="cu124" # Use latest stable supported by PyTorch
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            if [ "$CUDA_MINOR" -ge 4 ]; then
                CUDA_TAG="cu124"
            else
                CUDA_TAG="cu121"
            fi
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            if [ "$CUDA_MINOR" -ge 8 ]; then
                CUDA_TAG="cu118"
            else
                CUDA_TAG="cu118"
            fi
        else
            CUDA_TAG="cpu"
        fi
        
        if [ "$CUDA_TAG" = "cpu" ]; then
            echo "CUDA version $CUDA_VERSION may not be fully supported. Defaulting to CPU or standard install."
            EXTRA_INDEX=""
        else
            echo "Installing PyTorch with CUDA $CUDA_TAG support..."
            EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/$CUDA_TAG"
        fi
    else
        echo "No CUDA detected. Installing CPU version..."
        EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
    fi

    $PIP_CMD install -U \
        $INSTALL_ARGS \
        'torch' $EXTRA_INDEX
fi

# --- Install common packages ---
echo "Installing dependencies from requirements.txt..."
$PIP_CMD install \
    $INSTALL_ARGS \
    -r requirements.txt

# --- NEW: Install the current project in editable mode ---
echo "Installing project in editable mode..."
$PIP_CMD install \
    $INSTALL_ARGS \
    -e . 

echo ""
echo "✅ Installation complete."
echo "To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"
