import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """A PyTorch Dataset that loads sequences from a memory-mapped .bin file."""

    def __init__(self, data_path, sequence_length, dtype=np.uint16):
        # Memory-map the file
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.sequence_length = sequence_length

    def __len__(self):
        # Return the total number of possible sequences
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get a single sequence and its target
        x = torch.from_numpy(
            self.data[idx : idx + self.sequence_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.sequence_length].astype(np.int64)
        )
        return x, y
