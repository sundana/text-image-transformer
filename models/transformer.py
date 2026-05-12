import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, heads, T, d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.size()

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for text classification.

    Args:
        vocab_size:  Size of the token vocabulary.
        num_classes: Number of output classes.
        d_model:     Embedding / hidden dimension (default 128).
        num_heads:   Number of attention heads (default 4).
        num_layers:  Number of encoder blocks (default 2).
        d_ff:        Feed-forward inner dimension (default 256).
        max_len:     Maximum sequence length (default 512).
        dropout:     Dropout probability (default 0.1).
        pad_idx:     Padding token index for masking (default 0).
        pool:        Pooling strategy — "cls" uses the first token,
                     "mean" averages over non-padding tokens (default "mean").
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        pool: str = "mean",
    ):
        super().__init__()
        assert pool in ("cls", "mean"), "pool must be 'cls' or 'mean'"

        self.pad_idx = pad_idx
        self.pool = pool
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 1, T) — broadcast over heads and query positions
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token ids of shape (B, T).
        Returns:
            Logits of shape (B, num_classes).
        """
        mask = self._make_padding_mask(x)

        out = self.pos_enc(self.embedding(x))  # (B, T, d_model)

        for block in self.encoder:
            out = block(out, mask)

        if self.pool == "cls":
            pooled = out[:, 0]
        else:
            # mean over non-padding positions
            pad_mask = (x != self.pad_idx).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (out * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)


# Quick smoke-test
if __name__ == "__main__":
    VOCAB_SIZE = 10_000
    NUM_CLASSES = 2
    BATCH = 8
    SEQ_LEN = 64

    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=512,
        dropout=0.1,
        pool="mean",
    )

    dummy = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
    dummy[0, 50:] = 0  # simulate padding in first sample

    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
