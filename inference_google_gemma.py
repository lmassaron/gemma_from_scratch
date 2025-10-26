"""inference"""

from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from gemma_scratch.model import Gemma3Model, load_weights_into_gemma
from gemma_scratch.config import GEMMA3_CONFIG_270M
from gemma_scratch.tokenizer import gemma_tokenizer


def stream_next_token(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.7,
    repetition_penalty=1.2,
):
    """
    Generates text with temperature sampling and repetition penalty.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get the logits from the model
            logits = model(token_ids)[0][:, -1]

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                unique_tokens = torch.unique(token_ids)
                logits[:, unique_tokens] /= repetition_penalty

            # Apply temperature
            if temperature > 0.0:
                logits /= temperature

            # Get probabilities with softmax
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


def generate(
    sentence,
    model,
    tokenizer,
    device,
    max_new_tokens=200,
):
    """Encodes a sentence and yields the decoded generated tokens."""
    input_token_ids = tokenizer.encode(sentence)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    for token_tensor in stream_next_token(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.encode("<end_of_turn>")[-1],
        temperature=0.7,
        repetition_penalty=1.2,
    ):
        token_id_list = token_tensor.squeeze(0).tolist()
        yield tokenizer.decode(token_id_list)


if __name__ == "__main__":
    torch.manual_seed(123)
    print("Initializing the Gemma 3 Model")
    model = Gemma3Model(GEMMA3_CONFIG_270M)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"Total number of unique parameters: {total_params_normalized:,}")

    print("\nDownloading and Loading Weights")
    repo_id = "google/gemma-3-270m-it"
    local_dir = Path(repo_id).parts[-1]
    print(f"-> Starting download of '{repo_id}' from Hugging Face Hub")
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights_dict = load_file(weights_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_weights_into_gemma(model, GEMMA3_CONFIG_270M, weights_dict)
    model.to(device)
    del weights_dict

    print("-> Starting model compilation (JIT) with torch.compile()")
    model = torch.compile(model)
    model.eval()

    print("-> Warming up the model")
    warmup_input = torch.tensor([[0]], device=device)  # A dummy input
    with torch.no_grad():
        model(warmup_input)

    print("\n-> Text generation")
    test_sentences = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "A boy told his sister a bedtime story about a flying cat",
        "The kids sat in a circle while Uncle narrated a story about a brave knight",
        "Dad was telling the kids an adventure tale about a pirate ship",
    ]

    for k, test_sentence in enumerate(test_sentences):
        print(f'\n** Test {k:2d}. input sentence: "{test_sentence}"\n')
        for token_str in generate(
            test_sentence, model, gemma_tokenizer, device, max_new_tokens=200
        ):
            print(token_str, end="", flush=True)
        print()
