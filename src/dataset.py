"""
PyTorch Dataset and DataLoader module for AI Music Generation.
Handles sequence batching, train/val splitting, and tensor conversions.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np


class MusicDataset(Dataset):
    """
    Dataset wrapper for tokenized musical sequences and target notes.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    train_split: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Splits data into train and validation sets and returns PyTorch DataLoaders.
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    split_idx = int(num_samples * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = MusicDataset(X[train_indices], y[train_indices])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = None
    if len(val_indices) > 0:
        val_dataset = MusicDataset(X[val_indices], y[val_indices])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, val_loader
