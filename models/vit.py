"""Vision Transformer (ViT) — patch embedding + CLS token + reusable encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerEncoderBlock


class PatchEmbedding(nn.Module):
    """Image → patches via strided Conv2d, then flatten to (B, N, d_model)."""

    def __init__(self, in_channels: int, patch_size: int, d_model: int, img_size: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, d_model, H/p, W/p) → (B, N, d_model)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Args:
        img_size:     Side length of the (square) input image.
        patch_size:   Side length of each square patch.
        in_channels:  Number of image channels (3 for RGB).
        num_classes:  Number of output classes.
        d_model:      Token embedding dim.
        num_heads:    Number of attention heads.
        num_layers:   Number of transformer encoder blocks.
        d_ff:         Feed-forward inner dim.
        dropout:      Dropout probability throughout.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
        d_ff: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model, img_size)
        n_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                                   # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                            # (B, N+1, d_model)
        x = self.dropout(x + self.pos_embed)

        for block in self.encoder:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


if __name__ == "__main__":
    model = ViT(img_size=32, patch_size=4, num_classes=10)
    dummy = torch.randn(8, 3, 32, 32)
    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Patches: {model.patch_embed.num_patches}")
