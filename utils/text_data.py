"""IMDb data utilities: unzip → clean → tokenize → vocab → encode → DataLoader."""

from __future__ import annotations

import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1

_HTML_RE = re.compile(r"<[^>]+>")
_NONWORD_RE = re.compile(r"[^a-z0-9'\s]")


def clean_text(s: str) -> str:
    s = s.lower()
    s = _HTML_RE.sub(" ", s)
    s = _NONWORD_RE.sub(" ", s)
    return s


def tokenize(s: str) -> list[str]:
    return clean_text(s).split()


def extract_imdb_csv(zip_path: str | Path, out_dir: str | Path) -> Path:
    """Unzip IMDb dataset and return path to the CSV file."""
    zip_path, out_dir = Path(zip_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))

    return out_dir / csv_name


def build_vocab(token_lists: list[list[str]], max_vocab: int = 20_000) -> dict[str, int]:
    """Vocabulary from a list of token lists. Reserves PAD=0, UNK=1."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    most_common = counter.most_common(max_vocab - 2)
    stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for tok, _ in most_common:
        stoi[tok] = len(stoi)
    return stoi


def encode(tokens: list[str], stoi: dict[str, int], max_len: int) -> list[int]:
    ids = [stoi.get(t, UNK_IDX) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids


class IMDbDataset(Dataset):
    def __init__(self, encoded: list[list[int]], labels: list[int]):
        self.x = torch.tensor(encoded, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def prepare_imdb_loaders(
    zip_path: str | Path,
    out_dir: str | Path = "data/imdb",
    max_len: int = 256,
    max_vocab: int = 20_000,
    test_size: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 2,
):
    """Returns (train_loader, test_loader, stoi, raw_test_texts, raw_test_labels)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    csv_path = extract_imdb_csv(zip_path, out_dir)
    df = pd.read_csv(csv_path)

    # Column names: 'review' + 'sentiment' (positive/negative)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    texts = df["review"].astype(str).tolist()
    labels = df["label"].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    train_tokens = [tokenize(t) for t in train_texts]
    test_tokens = [tokenize(t) for t in test_texts]

    stoi = build_vocab(train_tokens, max_vocab=max_vocab)

    train_enc = [encode(t, stoi, max_len) for t in train_tokens]
    test_enc = [encode(t, stoi, max_len) for t in test_tokens]

    train_ds = IMDbDataset(train_enc, train_labels)
    test_ds = IMDbDataset(test_enc, test_labels)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, stoi, test_texts, test_labels


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    zip_path = here / "data" / "imdb-dataset-of-50k-movie-reviews.zip"
    train_loader, test_loader, stoi, _, _ = prepare_imdb_loaders(
        zip_path, out_dir=here / "data" / "imdb", max_len=128, batch_size=32, num_workers=0,
    )
    print(f"Vocab size: {len(stoi)}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    xb, yb = next(iter(train_loader))
    print(f"Batch x: {xb.shape} dtype={xb.dtype}, y: {yb.shape} dtype={yb.dtype}")
    print(f"Label distribution in batch: {yb.bincount().tolist()}")
