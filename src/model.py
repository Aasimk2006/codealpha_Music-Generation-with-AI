"""
Deep Learning Models for AI Music Generation
Includes:
1. MusicLSTM: Deep Multi-layer Recurrent Neural Network with Pitch Embeddings and Dropout.
2. MusicGRU: Gated Recurrent Unit alternative for efficient temporal sequence learning.
3. MusicGAN: Generative Adversarial Network architecture (Generator + Discriminator) for musical sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class MusicLSTM(nn.Module):
    """
    Multi-layer LSTM Neural Network with Embedding Layer for Music Generation.
    Maps musical token sequences to next-token probability distributions.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(MusicLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Dense musical representation layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Regularization & projection head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim // 2, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len)
            hidden: Optional (h_0, c_0) hidden state tuple
        Returns:
            logits: Tensor of shape (batch_size, vocab_size) for next-token prediction
            hidden: updated hidden state tuple
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Take output of the last sequence step
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        dropped = self.dropout(last_step)
        dense = self.relu(self.fc1(dropped))
        logits = self.out(dense)        # (batch_size, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes hidden and cell states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


class MusicGRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) model for music sequence modeling.
    Faster execution with comparable expressive capacity.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(MusicGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x)
        gru_out, hidden = self.gru(embeds, hidden)
        last_step = gru_out[:, -1, :]
        logits = self.fc(self.dropout(last_step))
        return logits, hidden


# -------------------------------------------------------------
# Optional GAN Module for Music Sequence Generation
# -------------------------------------------------------------
class MusicGenerator(nn.Module):
    """
    GAN Generator: Takes latent noise vector z and generates a sequence of note embeddings.
    """
    def __init__(self, latent_dim: int, seq_len: int, vocab_size: int, hidden_dim: int = 256):
        super(MusicGenerator, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (batch, latent_dim) -> (batch, hidden_dim)
        h = F.relu(self.fc(z))
        # Repeat across sequence length
        h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(h_seq)
        # Logits over vocabulary for each time step
        logits = self.out(lstm_out)  # (batch, seq_len, vocab_size)
        return logits


class MusicDiscriminator(nn.Module):
    """
    GAN Discriminator: Evaluates whether a musical note sequence is real or generated.
    """
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int = 256):
        super(MusicDiscriminator, self).__init__()
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        # x_onehot: (batch, seq_len, vocab_size)
        embeds = F.relu(self.embedding(x_onehot))
        lstm_out, _ = self.lstm(embeds)
        # Classify sequence authenticity
        validity = torch.sigmoid(self.fc(lstm_out[:, -1, :]))
        return validity


def create_model(
    model_type: str = "lstm",
    vocab_size: int = 316,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3
) -> nn.Module:
    """
    Factory helper to instantiate deep learning music models.
    """
    m_type = model_type.lower()
    if m_type == "lstm":
        return MusicLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif m_type == "gru":
        return MusicGRU(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lstm' or 'gru'.")
