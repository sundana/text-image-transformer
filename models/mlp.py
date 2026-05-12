from typing import Optional

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for text classification.

    Pipeline:
        Input (B, T)
        → Embedding (B, T, embed_dim)
        → Mean pooling over non-padding tokens (B, embed_dim)
        → MLP hidden layers
        → Output logits (B, num_classes)

    Args:
        vocab_size:   Size of the token vocabulary.
        num_classes:  Number of output classes.
        embed_dim:    Embedding dimension (default 128).
        hidden_dims:  List of hidden layer sizes (default [256, 128]).
        dropout:      Dropout probability applied between layers (default 0.3).
        pad_idx:      Padding token index for masking (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token ids of shape (B, T).
        Returns:
            Logits of shape (B, num_classes).
        """
        emb = self.embedding(x)                              # (B, T, embed_dim)

        # mean pooling — ignore padding tokens
        pad_mask = (x != self.pad_idx).float().unsqueeze(-1) # (B, T, 1)
        pooled = (emb * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)

        return self.mlp(pooled)


# Quick smoke-test
if __name__ == "__main__":
    VOCAB_SIZE = 10_000
    NUM_CLASSES = 2
    BATCH = 8
    SEQ_LEN = 64

    model = MLPClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        hidden_dims=[256, 128],
        dropout=0.3,
    )

    dummy = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
    dummy[0, 50:] = 0  # simulate padding

    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
