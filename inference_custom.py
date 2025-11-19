"""inference"""

import torch
import tiktoken
import argparse
from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM

enc = tiktoken.get_encoding("gpt2")


def generate(
    sentence, model, tokenizer, device, max_new_tokens=200, temperature=1.0, top_k=None
):
    context = torch.tensor(
        tokenizer.encode_ordinary(sentence), device=device
    ).unsqueeze(dim=0)

    with torch.no_grad():
        y = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eot_token,
        )

    return tokenizer.decode(y.squeeze().tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Gemma model."
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="./models/best_model_params_01.pt",
        help="Path to the saved model parameters (.pt file). Defaults to './models/best_model_params_01.pt'.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls randomness. Lower is more deterministic.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Sample from the top K most likely tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set a fixed random seed for reproducibility. If not set, generation is random.",
    )

    args = parser.parse_args()

    # Set determinism if a seed is provided
    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    else:
        print("No seed provided. Using random generation.")
        
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

    # Load the checkpoint
    checkpoint = torch.load(
        model_params_path, map_location=torch.device(device), weights_only=True
    )

    # Fix the keys if the model has been compiled (remove "_orig_mod." prefix)
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    # Load the fixed state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_sentences = [
        "Once upon a time there was a pumpkin.",
        # "A little girl went to the woods",
        # "A boy told his sister a bedtime story about a flying cat",
        # "The kids sat in a circle while Uncle narrated a story about a brave knight",
        # "Dad was telling the kids an adventure tale about a pirate ship",
    ]

    for k, test_sentence in enumerate(test_sentences):
        print(f"{k + 1:2d}. input sentence: {test_sentence}")
        generated = generate(
            test_sentence,
            model,
            enc,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(generated)
        print(f"\n{'-' * 64}\n")
