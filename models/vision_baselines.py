"""Vision baselines for CIFAR-10: ImageMLP, SimpleCNN, SmallResNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageMLP(nn.Module):
    """Flatten → MLP. Strong-ish baseline for tiny images."""

    def __init__(self, in_channels: int = 3, img_size: int = 32, num_classes: int = 10,
                 hidden_dims: list[int] | None = None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        in_dim = in_channels * img_size * img_size

        layers: list[nn.Module] = [nn.Flatten()]
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """3 conv blocks (Conv-BN-ReLU-Pool) → GAP → Linear."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                      # 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                      # 16→8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                              # 8→1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Simple ResNet block with optional 1x1 shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SmallResNet(nn.Module):
    """3-stage ResNet (channels 32→64→128) — compact, ~0.3M params."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(32, 32),  BasicBlock(32, 32))
        self.layer2 = nn.Sequential(BasicBlock(32, 64, stride=2),  BasicBlock(64, 64))
        self.layer3 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    dummy = torch.randn(8, 3, 32, 32)
    for cls in (ImageMLP, SimpleCNN, SmallResNet):
        m = cls()
        out = m(dummy)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__:12s}  out={tuple(out.shape)}  params={n_params:,}")
