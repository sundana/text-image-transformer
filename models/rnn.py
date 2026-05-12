import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    Vanilla RNN for text classification.

    Pipeline:
        Input (B, T)
        → Embedding (B, T, embed_dim)
        → RNN (num_layers stacked, optional bidirectional)
        → Pooling: last hidden state or mean over all timesteps
        → Dropout → Linear → Output logits (B, num_classes)

    Args:
        vocab_size:    Size of the token vocabulary.
        num_classes:   Number of output classes.
        embed_dim:     Embedding dimension (default 128).
        hidden_dim:    RNN hidden state size (default 128).
        num_layers:    Number of stacked RNN layers (default 2).
        bidirectional: Use bidirectional RNN (default True).
        dropout:       Dropout between RNN layers and before classifier (default 0.3).
        pad_idx:       Padding token index (default 0).
        pool:          Pooling strategy — "last" uses the final hidden state,
                       "mean" averages over all timesteps (default "last").
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pool: str = "last",
    ):
        super().__init__()
        assert pool in ("last", "mean"), "pool must be 'last' or 'mean'"

        self.pad_idx = pad_idx
        self.pool = pool
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * directions, num_classes)

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
        out, hidden = self.rnn(emb)        # out: (B, T, hidden_dim * directions)
                                           # hidden: (num_layers * directions, B, hidden_dim)

        if self.pool == "last":
            directions = 2 if self.bidirectional else 1
            # Take the last layer's hidden state(s) and concat directions
            pooled = hidden[-directions:].transpose(0, 1).contiguous()
            pooled = pooled.view(pooled.size(0), -1)   # (B, hidden_dim * directions)
        else:
            # Mean over non-padding timesteps
            pad_mask = (x != self.pad_idx).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (out * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)

        return self.classifier(self.dropout(pooled))


# Quick smoke-test
if __name__ == "__main__":
    VOCAB_SIZE = 10_000
    NUM_CLASSES = 2
    BATCH = 8
    SEQ_LEN = 64

    model = RNNClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
        pool="last",
    )

    dummy = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
    dummy[0, 50:] = 0  # simulate padding

    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
