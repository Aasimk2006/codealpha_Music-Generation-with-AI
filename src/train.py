"""
Training Pipeline for AI Music Generation
Trains the Deep MusicLSTM / MusicGRU model with loss tracking,
validation evaluation, learning rate scheduling, and checkpointing.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from src.model import MusicLSTM, MusicGRU, create_model
from src.dataset import get_data_loaders
from src.preprocess import load_preprocessed_data, PROCESSED_DIR


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_model(
    model_type: str = "lstm",
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    device: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Dict[str, Any]:
    """
    Trains the music generation neural network.
    
    Args:
        model_type: "lstm" or "gru"
        epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        learning_rate: Initial learning rate for AdamW
        embedding_dim: Dimension of pitch embedding vectors
        hidden_dim: Hidden dimension of LSTM layers
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        device: "cuda", "cpu", or auto-detect
        checkpoint_path: Custom path to save the best model weights
        progress_callback: Callback fn(epoch, total_epochs, train_loss, val_loss)
    """
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using compute device: {dev}")

    # Load dataset
    X, y, note_to_int, int_to_note, metadata, _ = load_preprocessed_data()
    vocab_size = len(note_to_int)
    print(f"Dataset loaded: {len(X)} sequences | Vocab size: {vocab_size}")

    train_loader, val_loader = get_data_loaders(
        X, y, batch_size=batch_size, train_split=0.9, shuffle=True
    )

    # Initialize model
    model = create_model(
        model_type=model_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "epochs": []}
    save_file = checkpoint_path or (CHECKPOINT_DIR / f"best_music_{model_type}.pt")

    print(f"\n--- Starting Training ({model_type.upper()}) for {epochs} Epochs ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(dev), targets.to(dev)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % max(1, total_batches // 5) == 0:
                step_loss = running_loss / (batch_idx + 1)
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{total_batches}] - Loss: {step_loss:.4f}")

        train_loss = running_loss / total_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(dev), targets.to(dev)
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start
        print(
            f"--> Epoch {epoch:02d}/{epochs:02d} Complete: "
            f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} "
            f"({epoch_duration:.1f}s)"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_config": {
                    "model_type": model_type,
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "sequence_length": metadata.get("sequence_length", 32)
                }
            }, save_file)
            print(f"    * Saved new best model to {save_file.name} (Val Loss: {val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\n--- Training Complete in {total_time:.1f} seconds! ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Save training history JSON
    history_file = CHECKPOINT_DIR / "history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint_file": str(save_file)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train AI Music Generator")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"], help="Model type")
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
