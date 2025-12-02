import argparse
import os
import numpy as np
from gemma_scratch.tokenizer import get_gpt2_tokenizer


def count_tokens(data_dir, batch_size, sequence_length):
    """
    Counts the tokens for the train and validation splits and computes the tokens by batch.
    """
    tokenizer = get_gpt2_tokenizer()

    for split in ["train", "val"]:
        data_path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(data_path):
            # Memory-map the file
            data = np.memmap(data_path, dtype=np.uint16, mode="r")
            num_tokens = len(data)
            print(f"Number of tokens in {split} split: {num_tokens}")

            num_sequences = len(data) - sequence_length
            num_batches = num_sequences // batch_size
            tokens_per_batch = batch_size * sequence_length
            print(f"Number of sequences in {split} split: {num_sequences}")
            print(f"Number of batches in {split} split: {num_batches}")
            print(f"Tokens per batch in {split} split: {tokens_per_batch}")
            print("-" * 20)
        else:
            print(f"Could not find {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in a dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tinystories_data",
        help="Directory with train.bin and val.bin.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Context window length (block size).",
    )
    args = parser.parse_args()

    count_tokens(args.data_dir, args.batch_size, args.sequence_length)
