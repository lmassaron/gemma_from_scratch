import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from gemma_scratch.tokenizer import get_gpt2_tokenizer

enc = get_gpt2_tokenizer()


def process(example):
    """Encoding text."""
    # encode_ordinary gives a clean list of integers
    ids = enc.encode_ordinary(example["text"])

    # Append the special <|endoftext|> token (ID 50256) to the end
    ids.append(enc.eot_token)

    # Verify the length is correct (now +1 compared to before)
    out = {"ids": ids, "len": len(ids)}

    return out


def prepare_data(dataset_name, output_dir="."):
    """
    Preparing a large text dataset for machine learning model training.

    Args:
        dataset_name (str): The name of the dataset to load from the Hugging Face Hub.
        output_dir (str, optional): The directory to save the output files. Defaults to ".".
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    data = load_dataset(dataset_name)
    print(f"Found splits: {list(data.keys())}")

    # Process and save the dataset
    if not os.path.exists(os.path.join(output_dir, "train.bin")):
        tokenized = data.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=os.cpu_count(),  # returns the number of available CPUs for tokenization on multiple cores
        )
        # Concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            # calculates the exact number of tokens that will be in the final binary file
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            if split == "validation":
                split = "val"
            print(f"Tokenizing {split} split - containing {arr_len} total tokens")
            filename = os.path.join(output_dir, f"{split}.bin")
            # (can do since enc.max_token_value == 50256 is < 2**16)
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024  # a hardcoded number that controls the granularity of the writing process
            # Caveaty: for very large datasets (e.g., hundreds of GB), each of the 1024 shards could be very large, potentially causing memory issues.
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                # Processes the entire dataset by breaking it into chunks, which it calls "shards"
                # The dset.shard() function splits the dataset into total_batches (1024) pieces
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                )
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a dataset for training.")
    parser.add_argument(
        "dataset_name",
        nargs="?",
        type=str,
        default="roneneldan/TinyStories",
        help="The name of the dataset to load from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tinystories_data",
        help="The directory to save the output files.",
    )
    args = parser.parse_args()
    prepare_data(args.dataset_name, args.output_dir)
