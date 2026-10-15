"""
Music Sequence Generator Module
Uses the trained deep learning model to generate new musical note sequences
using temperature-scaled autoregressive sampling and top-k filtering.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from src.model import MusicLSTM, MusicGRU, create_model
from src.preprocess import load_preprocessed_data, PROCESSED_DIR


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def load_generator_model(
    checkpoint_file: Optional[Path] = None,
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, int], Dict[int, str]]:
    """
    Loads the trained model weights and configuration.
    """
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    ckpt_path = checkpoint_file or (CHECKPOINT_DIR / "best_music_lstm.pt")
    if not ckpt_path.exists():
        # Fallback to GRU if LSTM not found
        alt_path = CHECKPOINT_DIR / "best_music_gru.pt"
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. Please train a model first!"
            )

    checkpoint = torch.load(ckpt_path, map_location=dev)
    config = checkpoint["model_config"]

    # Load vocabulary
    with open(PROCESSED_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    note_to_int = vocab_data["note_to_int"]
    int_to_note = {int(k): v for k, v in vocab_data["int_to_note"].items()}

    model = create_model(
        model_type=config.get("model_type", "lstm"),
        vocab_size=config["vocab_size"],
        embedding_dim=config.get("embedding_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3)
    ).to(dev)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {config.get('model_type', 'lstm').upper()} model from {ckpt_path.name} (Val Loss: {checkpoint.get('val_loss', 'N/A')})")

    return model, config, note_to_int, int_to_note


def sample_with_temperature_and_topk(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 20
) -> int:
    """
    Applies temperature scaling and top-k filtering, then samples from categorical distribution.
    
    - temperature < 1.0: structured, conservative, predictable
    - temperature > 1.0: creative, exploratory, jazz-like
    - top_k: keeps only top K candidates to eliminate noise
    """
    # Scale logits by temperature
    scaled_logits = logits / max(1e-4, temperature)

    if top_k > 0:
        values, indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        min_value = values[:, -1].unsqueeze(-1)
        scaled_logits = torch.where(
            scaled_logits < min_value,
            torch.full_like(scaled_logits, -float("inf")),
            scaled_logits
        )

    probs = F.softmax(scaled_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1).item()
    return next_token_idx


def generate_music_sequence(
    num_notes: int = 100,
    temperature: float = 0.85,
    top_k: int = 15,
    seed_sequence: Optional[List[str]] = None,
    checkpoint_file: Optional[Path] = None,
    device: Optional[str] = None
) -> List[str]:
    """
    Autoregressively generates a new sequence of musical notes and chords.
    
    Args:
        num_notes: Number of notes to generate
        temperature: Sampling temperature (0.5 to 1.5)
        top_k: Top-K filtering cutoff
        seed_sequence: Optional initial sequence of note strings
        checkpoint_file: Path to model checkpoint
        device: Device to run inference on
    Returns:
        List of generated note strings (e.g. ['C4', 'E4.G4', 'REST', 'D4', ...])
    """
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, config, note_to_int, int_to_note = load_generator_model(checkpoint_file, device=dev)
    seq_length = config.get("sequence_length", 32)

    # 1. Determine primer seed sequence
    if seed_sequence and len(seed_sequence) >= seq_length:
        current_seq = [
            note_to_int.get(note, 0)
            for note in seed_sequence[-seq_length:]
        ]
    else:
        # Load random sequence from dataset
        X = np.load(PROCESSED_DIR / "X.npy")
        random_idx = np.random.randint(0, len(X))
        current_seq = list(X[random_idx])

    generated_indices = []

    print(f"Generating {num_notes} musical events (temp={temperature}, top_k={top_k})...")
    
    with torch.no_grad():
        for step in range(num_notes):
            input_tensor = torch.tensor([current_seq[-seq_length:]], dtype=torch.long, device=dev)
            logits, _ = model(input_tensor)
            
            next_idx = sample_with_temperature_and_topk(
                logits, temperature=temperature, top_k=top_k
            )

            current_seq.append(next_idx)
            generated_indices.append(next_idx)

    # Convert indices back to note tokens
    generated_notes = [
        int_to_note.get(idx, "C4")
        for idx in generated_indices
    ]

    # Clean tokens: remove song boundary tokens
    clean_notes = [n for n in generated_notes if n != "<SEP>"]

    print(f"Generated {len(clean_notes)} musical tokens successfully!")
    return clean_notes


if __name__ == "__main__":
    notes = generate_music_sequence(num_notes=50, temperature=0.8)
    print("Sample generated notes:", notes[:15])
