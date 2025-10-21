VENV_NAME=".venv"
PYTHON_VERSION="3.12"

# Create the virtual environment
uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --seed

uv pip install \
    --python "$VENV_NAME/bin/python" \
    'torch' --extra-index-url https://download.pytorch.org/whl/cu121 \
    'datasets' \
    'tiktoken' \
    'tokenizers' \
    'safetensors' \
    'numpy' \
    'tqdm' \
    'tensorboard' \
    'matplotlib'

echo "To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"