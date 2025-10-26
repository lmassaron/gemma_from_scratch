"""inference"""

import torch
import tiktoken
import argparse
from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM

enc = tiktoken.get_encoding("gpt2")


def generate(sentence, model, tokenizer, device, max_new_tokens=200):
    context = torch.tensor(
        tokenizer.encode_ordinary(sentence), device=device
    ).unsqueeze(dim=0)

    with torch.no_grad():
        y = model.generate(context, max_new_tokens)

    return tokenizer.decode(y.squeeze().tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Gemma model."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=str,
        default="./models/best_model_params_01.pt",
        help="Path to the saved model parameters (.pt file). Defaults to './models/best_model_params_01.pt'.",
    )
    args = parser.parse_args()

    torch.manual_seed(123)
    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)

    # Set the device (mps for Apple Silicon, cuda for NVIDIA, cpu as fallback)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    model_params_path = args.model_path
    model.load_state_dict(
        torch.load(
            model_params_path, map_location=torch.device(device), weights_only=True
        )
    )

    model.to(device)
    model.eval()

    test_sentences = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "A boy told his sister a bedtime story about a flying cat",
        "The kids sat in a circle while Uncle narrated a story about a brave knight",
        "Dad was telling the kids an adventure tale about a pirate ship",
    ]

    for k, test_sentence in enumerate(test_sentences):
        print(f"{k:2d}. input sentence: {test_sentence}")
        generated = generate(test_sentence, model, enc, device, max_new_tokens=200)
        print(generated)
        print("-" * 64)
