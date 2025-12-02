"""A tokenizer for the Gemma-3 model and a utility for applying a chat template"""

import sys
from pathlib import Path
import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# --- Constants ---
CHOOSE_MODEL = "270m"
REPO_ID = f"google/gemma-3-{CHOOSE_MODEL}-it"
TOKENIZER_FILE = "tokenizer.json"
LOCAL_DIR = Path(REPO_ID).parts[-1]

# Special tokens used by the Gemma-3 model
EOS_TOKEN = "<end_of_turn>"
START_USER_TOKEN = "<start_of_turn>user\n"
START_MODEL_TOKEN = "\n<start_of_turn>model\n"


class GemmaTokenizer:
    """
    A tokenizer for the Gemma-3 model that wraps the tokenizers library.

    This class loads a tokenizer from a file and provides methods for encoding
    and decoding text, while also storing important token IDs like the end-of-turn
    token.
    """

    def __init__(self, tokenizer_file_path: Path):
        """
        Initializes the tokenizer from the given file path.

        Args:
            tokenizer_file_path (Path): The local path to the tokenizer.json file.
        """
        self._tok = Tokenizer.from_file(str(tokenizer_file_path))

        # Get the integer ID for the end-of-turn token
        self.eos_token_id = self._tok.token_to_id(EOS_TOKEN)
        self.pad_token_id = self.eos_token_id

    @property
    def vocab_size(self) -> int:
        """
        Returns the total number of unique tokens in the vocabulary.
        """
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string of text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of corresponding token IDs.
        """
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        return self._tok.decode(ids, skip_special_tokens=False)


def apply_chat_template(user_text: str) -> str:
    """
    Applies the Gemma-3 chat template to a user's message.

    This function formats the input text according to the model's specified
    chat structure, indicating the start of the user's turn and prompting
    the model for a response.

    Args:
        user_text (str): The user's input message.

    Returns:
        str: The formatted string ready for the model.
    """
    return f"{START_USER_TOKEN}{user_text}{EOS_TOKEN}{START_MODEL_TOKEN}"


def download_tokenizer_if_needed(repo_id: str, local_dir: Path) -> Path:
    """
    Downloads the tokenizer file from Hugging Face Hub if it doesn't exist locally.
    In case you have no permission to load it, it falls back to

    Args:
        repo_id (str): The Hugging Face repository ID to download from.
        local_dir (Path): The local directory to save the file in.

    Returns:
        Path: The path to the local tokenizer file.
    """
    tokenizer_path = Path(local_dir) / Path(TOKENIZER_FILE)
    if not tokenizer_path.exists():
        print(f"'{tokenizer_path}' not found. Downloading...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=TOKENIZER_FILE,
                local_dir=local_dir,
            )
            print("Download complete.")
        except (IOError, ValueError) as e:
            print(f"Warning: Failed to download tokenizer.json: {e}", file=sys.stderr)
            print("Attemping to load it from a fallback repository")
            try:
                fallback_repo_id = "lmassaron/gemma-3-4b-finsentiment"
                hf_hub_download(
                    repo_id=fallback_repo_id,
                    filename=TOKENIZER_FILE,
                    local_dir=local_dir,
                )
                print("Download complete.")
            except (IOError, ValueError) as e:
                print(
                    f"Warning: Failed to download tokenizer.json: {e}", file=sys.stderr
                )

    return tokenizer_path


def get_gemma_tokenizer():
    tokenizer_file = download_tokenizer_if_needed(REPO_ID, LOCAL_DIR)
    gemma_tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file)
    return gemma_tokenizer


def get_gpt2_tokenizer():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return gpt2_tokenizer


if __name__ == "__main__":
    # Instantiating a tokenizer class with the downloaded Gemma tokenizer.
    gemma_tokenizer = get_gemma_tokenizer()

    # Defining a sample message from a user.
    user_message = "What is the purpose of a tokenizer?"

    # Applying the model-specific chat template.
    # This formats the input correctly for the model to understand it's a user prompt.
    formatted_prompt = apply_chat_template(user_message)

    print("--- Chat Template ---")
    print(f"Original message: '{user_message}'")
    print(f"Formatted for model:\n'{formatted_prompt}'")
    print("-" * 25)

    # Encoding the formatted prompt into token IDs.
    encoded_ids = gemma_tokenizer.encode(formatted_prompt)

    print("--- Encoding ---")
    print(f"Vocabulary size: {gemma_tokenizer.vocab_size} tokens")
    print(f"Encoded token IDs: {encoded_ids}")
    print("-" * 25)

    # Decoding the token IDs back into text to verify.
    decoded_text = gemma_tokenizer.decode(encoded_ids)

    print("--- Decoding ---")
    print(f"Decoded text: '{decoded_text}'")
    print("-" * 25)

    # Comparing with another tokenizer like GPT-2's.
    print("--- Comparison with GPT-2 Tokenizer ---")
    gpt2_tokenizer = get_gpt2_tokenizer()
    gpt2_encoded = gpt2_tokenizer.encode(user_message)
    print(f"Same message encoded with GPT-2: {gpt2_encoded}")
    print("Notice how the token IDs and the length of the list can be different")
