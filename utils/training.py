"""Generic training & evaluation utilities reused across all models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)
    train_time: float = 0.0
    inference_time_ms: float = 0.0
    num_params: int = 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (y_pred, y_true) as 1-D LongTensors on CPU."""
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=-1).cpu())
        trues.append(yb)
    return torch.cat(preds), torch.cat(trues)


@torch.no_grad()
def time_inference(
    model: nn.Module, loader: DataLoader, device: torch.device, n_batches: int = 10
) -> float:
    """Average inference time per batch in milliseconds."""
    model.eval()
    it = iter(loader)
    # warmup
    try:
        xb, _ = next(it)
        model(xb.to(device))
    except StopIteration:
        return 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    count = 0
    for _ in range(n_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        model(xb.to(device))
        count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / max(count, 1)) * 1000.0


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device | str = "cuda",
    verbose: bool = True,
    name: str = "model",
) -> History:
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = History(num_params=count_parameters(model))

    start = time.perf_counter()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.test_loss.append(te_loss)
        history.test_acc.append(te_acc)
        if verbose:
            print(
                f"[{name}] epoch {ep:2d}/{epochs}  "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
                f"test loss={te_loss:.4f} acc={te_acc:.4f}"
            )
    history.train_time = time.perf_counter() - start
    history.inference_time_ms = time_inference(model, test_loader, device)
    if verbose:
        print(
            f"[{name}] done — params={history.num_params:,}  "
            f"train_time={history.train_time:.1f}s  "
            f"inference/batch={history.inference_time_ms:.2f}ms"
        )
    return history
