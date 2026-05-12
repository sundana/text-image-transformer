from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DClassifier(nn.Module):
    """
    TextCNN (Kim 2014) — parallel Conv1D with multiple kernel sizes for text classification.

    Pipeline:
        Input (B, T)
        → Embedding (B, T, embed_dim)
        → N parallel Conv1D with different kernel sizes
        → Max-over-time pooling per kernel → each (B, num_filters)
        → Concatenate → (B, num_filters * len(kernel_sizes))
        → Dropout → Linear → Output logits (B, num_classes)

    Args:
        vocab_size:    Size of the token vocabulary.
        num_classes:   Number of output classes.
        embed_dim:     Embedding dimension (default 128).
        num_filters:   Number of filters per kernel size (default 128).
        kernel_sizes:  List of convolution window sizes (default [2, 3, 4]).
        dropout:       Dropout probability before classifier (default 0.5).
        pad_idx:       Padding token index (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: Optional[list[int]] = None,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Each conv: input (B, embed_dim, T) → output (B, num_filters, T - k + 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

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
        emb = self.embedding(x)            # (B, T, embed_dim)
        emb = emb.transpose(1, 2)          # (B, embed_dim, T) — Conv1d expects (B, C, L)

        # conv → ReLU → max-over-time pooling
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                              # (B, num_filters, T - k + 1)
            pooled.append(F.max_pool1d(c, c.size(2)).squeeze(2))  # (B, num_filters)

        out = torch.cat(pooled, dim=1)     # (B, num_filters * len(kernel_sizes))
        return self.classifier(self.dropout(out))


# Quick smoke-test
if __name__ == "__main__":
    VOCAB_SIZE = 10_000
    NUM_CLASSES = 2
    BATCH = 8
    SEQ_LEN = 64

    model = CNN1DClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_filters=128,
        kernel_sizes=[2, 3, 4],
        dropout=0.5,
    )

    dummy = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
    dummy[0, 50:] = 0  # simulate padding

    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
