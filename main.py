"""End-to-end pipeline (mirror of transformer_assignment.ipynb).

Runs all 16 sections of the assignment, saves all plots, tables, and metrics
to ./output/ for inclusion in the laporan.

Usage:
    conda activate deep-learning
    python main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.transformer import TransformerClassifier
from models.mlp         import MLPClassifier
from models.cnn1d       import CNN1DClassifier
from models.rnn         import RNNClassifier
from models.vit         import ViT
from models.vision_baselines import ImageMLP, SimpleCNN, SmallResNet
from utils.text_data    import prepare_imdb_loaders
from utils.training     import fit, collect_predictions, count_parameters, History

# ──────────────────────────────────────────────────────────────── output dirs
OUT = ROOT / "output"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in [OUT, PLOTS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=120, bbox_inches="tight")
    plt.close()


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────── 1-2 device
banner("Section 1-2: Imports & device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch       : {torch.__version__}")
print(f"torchvision : {torchvision.__version__}")
print(f"numpy       : {np.__version__}")
print(f"device      : {device}")
if device.type == "cuda":
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

results: dict = {"text": {}, "vision": {}, "seq_len": {}, "patch_size": {}, "sr": {}}

# ──────────────────────────────────────────────────────────────── 3-4 text data
banner("Section 3-4: Load & preprocess IMDb")
IMDB_ZIP = ROOT / "data" / "imdb-dataset-of-50k-movie-reviews.zip"
MAX_LEN_MAIN = 256
BATCH_TEXT = 64
VOCAB_SIZE = 20_000

# Quick peek at raw CSV
with zipfile.ZipFile(IMDB_ZIP) as zf:
    csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
    zf.extractall(ROOT / "data" / "imdb")
raw_df = pd.read_csv(ROOT / "data" / "imdb" / csv_name)
print(f"IMDb samples: {len(raw_df):,}, label balance: {raw_df['sentiment'].value_counts().to_dict()}")

train_loader, test_loader, stoi, raw_test_texts, raw_test_labels = prepare_imdb_loaders(
    IMDB_ZIP, out_dir=ROOT / "data" / "imdb",
    max_len=MAX_LEN_MAIN, max_vocab=VOCAB_SIZE,
    test_size=0.2, batch_size=BATCH_TEXT, num_workers=0, seed=42,
)
print(f"Vocab {len(stoi):,}, train batches {len(train_loader):,}, test batches {len(test_loader):,}")

# Length distribution plot
review_lens = [len(t.split()) for t in raw_test_texts]
plt.figure(figsize=(8, 3))
plt.hist(review_lens, bins=60, color="steelblue", edgecolor="white")
plt.axvline(MAX_LEN_MAIN, color="crimson", linestyle="--", label=f"max_len={MAX_LEN_MAIN}")
plt.xlabel("review length (words)"); plt.ylabel("count")
plt.title("IMDb review length distribution"); plt.legend()
save_fig("imdb_length_distribution.png")

# ──────────────────────────────────────────────────────────────── 5-6 text models
banner("Section 5-6: Text models instantiation")
transformer_text = TransformerClassifier(VOCAB_SIZE, 2, d_model=128, num_heads=4,
                                         num_layers=2, d_ff=256, max_len=MAX_LEN_MAIN,
                                         dropout=0.1, pool="mean")
mlp_text  = MLPClassifier(VOCAB_SIZE, 2, embed_dim=128, hidden_dims=[256, 128])
cnn_text  = CNN1DClassifier(VOCAB_SIZE, 2, embed_dim=128, num_filters=128, kernel_sizes=[2, 3, 4])
rnn_text  = RNNClassifier(VOCAB_SIZE, 2, embed_dim=128, hidden_dim=128, num_layers=2, bidirectional=True)
for name, m in [("Transformer", transformer_text), ("MLP", mlp_text),
                ("CNN1D", cnn_text), ("RNN", rnn_text)]:
    print(f"  {name:12s}  params = {count_parameters(m):>10,}")

# ──────────────────────────────────────────────────────────────── 7 text training
banner("Section 7: Text training (10 epochs each)")
EPOCHS_TEXT = 10
text_histories: dict[str, History] = {}
for name, model in [("Transformer", transformer_text), ("MLP", mlp_text),
                    ("CNN1D", cnn_text), ("RNN", rnn_text)]:
    text_histories[name] = fit(model, train_loader, test_loader,
                               epochs=EPOCHS_TEXT, device=device, name=name)

# Plot text training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in text_histories.items():
    e = range(1, len(h.train_loss) + 1)
    axes[0].plot(e, h.train_loss, label=f"{name} train")
    axes[0].plot(e, h.test_loss, "--", label=f"{name} test")
    axes[1].plot(e, h.train_acc, label=f"{name} train")
    axes[1].plot(e, h.test_acc, "--", label=f"{name} test")
axes[0].set_title("Text — loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
axes[1].set_title("Text — accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(fontsize=8)
save_fig("text_training_curves.png")

# Save text histories
for name, h in text_histories.items():
    results["text"][name] = {
        "test_acc": float(h.test_acc[-1]),
        "test_loss": float(h.test_loss[-1]),
        "params": int(h.num_params),
        "train_time_s": round(h.train_time, 1),
        "inference_ms": round(h.inference_time_ms, 2),
        "train_acc_history": [float(x) for x in h.train_acc],
        "test_acc_history": [float(x) for x in h.test_acc],
        "train_loss_history": [float(x) for x in h.train_loss],
        "test_loss_history": [float(x) for x in h.test_loss],
    }

# ──────────────────────────────────────────────────────────────── 7.1 seq_len
banner("Section 7.1: Sequence length variation (5 epochs each)")
SEQ_LENS = [64, 128, 256, 512]
EPOCHS_SEQ = 5
for L in SEQ_LENS:
    print(f"\n--- max_len = {L} ---")
    tl, te, _, _, _ = prepare_imdb_loaders(
        IMDB_ZIP, out_dir=ROOT / "data" / "imdb", max_len=L, max_vocab=VOCAB_SIZE,
        test_size=0.2, batch_size=BATCH_TEXT, num_workers=0, seed=42,
    )
    m = TransformerClassifier(VOCAB_SIZE, 2, d_model=128, num_heads=4, num_layers=2,
                              d_ff=256, max_len=L, dropout=0.1, pool="mean")
    h = fit(m, tl, te, epochs=EPOCHS_SEQ, device=device, name=f"Trf-L{L}", verbose=False)
    results["seq_len"][L] = {
        "test_acc": float(h.test_acc[-1]),
        "train_time_s": round(h.train_time, 1),
        "params": int(h.num_params),
    }
    print(f"  test_acc={h.test_acc[-1]:.4f}  time={h.train_time:.1f}s")

# ──────────────────────────────────────────────────────────────── 8 vision data
banner("Section 8: Load CIFAR-10")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
test_tf  = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
cifar_train = CIFAR10(root=str(ROOT / "data"), train=True,  download=True, transform=train_tf)
cifar_test  = CIFAR10(root=str(ROOT / "data"), train=False, download=True, transform=test_tf)
BATCH_VISION = 128
vision_train_loader = DataLoader(cifar_train, batch_size=BATCH_VISION, shuffle=True,
                                 num_workers=0, pin_memory=True)
vision_test_loader  = DataLoader(cifar_test,  batch_size=BATCH_VISION, shuffle=False,
                                 num_workers=0, pin_memory=True)
CIFAR_CLASSES = cifar_train.classes
print(f"Train {len(cifar_train):,}, Test {len(cifar_test):,}, classes: {CIFAR_CLASSES}")

# Sample images per class
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
shown: dict[int, bool] = {}
for img, label in cifar_train:
    if label not in shown:
        ax = axes[label // 5, label % 5]
        img_show = img * torch.tensor(CIFAR_STD).view(3,1,1) + torch.tensor(CIFAR_MEAN).view(3,1,1)
        ax.imshow(img_show.permute(1,2,0).clamp(0,1).numpy())
        ax.set_title(CIFAR_CLASSES[label]); ax.axis("off")
        shown[label] = True
    if len(shown) == 10:
        break
plt.suptitle("CIFAR-10 — one sample per class")
save_fig("cifar10_samples.png")

# ──────────────────────────────────────────────────────────────── 9-10 vision models
banner("Section 9-10: Vision models instantiation")
vit          = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10,
                   d_model=192, num_heads=6, num_layers=6, d_ff=384, dropout=0.1)
image_mlp    = ImageMLP()
simple_cnn   = SimpleCNN()
small_resnet = SmallResNet()
for name, m in [("ViT", vit), ("ImageMLP", image_mlp),
                ("SimpleCNN", simple_cnn), ("SmallResNet", small_resnet)]:
    print(f"  {name:13s}  params = {count_parameters(m):>10,}")

# ──────────────────────────────────────────────────────────────── 11 vision training
banner("Section 11: Vision training (10 epochs each)")
EPOCHS_VISION = 10
vision_histories: dict[str, History] = {}
for name, model in [("ViT", vit), ("ImageMLP", image_mlp),
                    ("SimpleCNN", simple_cnn), ("SmallResNet", small_resnet)]:
    vision_histories[name] = fit(model, vision_train_loader, vision_test_loader,
                                 epochs=EPOCHS_VISION, device=device, name=name)

# Plot vision training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in vision_histories.items():
    e = range(1, len(h.train_loss) + 1)
    axes[0].plot(e, h.train_loss, label=f"{name} train")
    axes[0].plot(e, h.test_loss, "--", label=f"{name} test")
    axes[1].plot(e, h.train_acc, label=f"{name} train")
    axes[1].plot(e, h.test_acc, "--", label=f"{name} test")
axes[0].set_title("Vision — loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
axes[1].set_title("Vision — accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(fontsize=8)
save_fig("vision_training_curves.png")

for name, h in vision_histories.items():
    results["vision"][name] = {
        "test_acc": float(h.test_acc[-1]),
        "test_loss": float(h.test_loss[-1]),
        "params": int(h.num_params),
        "train_time_s": round(h.train_time, 1),
        "inference_ms": round(h.inference_time_ms, 2),
        "train_acc_history": [float(x) for x in h.train_acc],
        "test_acc_history": [float(x) for x in h.test_acc],
        "train_loss_history": [float(x) for x in h.train_loss],
        "test_loss_history": [float(x) for x in h.test_loss],
    }

# ──────────────────────────────────────────────────────────────── 11.1 patch_size
banner("Section 11.1: Patch size variation (5 epochs each)")
PATCH_SIZES = [2, 4, 8, 16]
EPOCHS_PATCH = 5
for p in PATCH_SIZES:
    print(f"\n--- patch_size = {p} ---")
    m = ViT(img_size=32, patch_size=p, in_channels=3, num_classes=10,
            d_model=192, num_heads=6, num_layers=6, d_ff=384, dropout=0.1)
    h = fit(m, vision_train_loader, vision_test_loader,
            epochs=EPOCHS_PATCH, device=device, name=f"ViT-p{p}", verbose=False)
    results["patch_size"][p] = {
        "num_patches": int(m.patch_embed.num_patches),
        "test_acc": float(h.test_acc[-1]),
        "train_time_s": round(h.train_time, 1),
        "params": int(h.num_params),
    }
    print(f"  patches={m.patch_embed.num_patches}  test_acc={h.test_acc[-1]:.4f}  time={h.train_time:.1f}s")

# ──────────────────────────────────────────────────────────────── 12 plots — variations
banner("Section 12: Variation plots")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
xs = list(results["seq_len"].keys())
axes[0].bar([str(x) for x in xs], [results["seq_len"][x]["test_acc"] for x in xs], color="steelblue")
axes[0].set_xlabel("max_len"); axes[0].set_ylabel("test accuracy")
axes[0].set_title("Transformer (IMDb) — variasi sequence length"); axes[0].set_ylim(0.5, 1.0)
for i, x in enumerate(xs):
    axes[0].text(i, results["seq_len"][x]["test_acc"]+0.005,
                 f'{results["seq_len"][x]["test_acc"]:.3f}', ha="center")

ps = list(results["patch_size"].keys())
axes[1].bar([str(p) for p in ps], [results["patch_size"][p]["test_acc"] for p in ps], color="darkorange")
axes[1].set_xlabel("patch size"); axes[1].set_ylabel("test accuracy")
axes[1].set_title("ViT (CIFAR-10) — variasi patch size"); axes[1].set_ylim(0.2, 0.85)
for i, p in enumerate(ps):
    axes[1].text(i, results["patch_size"][p]["test_acc"]+0.005,
                 f'{results["patch_size"][p]["test_acc"]:.3f}', ha="center")
save_fig("variations.png")

# ──────────────────────────────────────────────────────────────── 13 confusion + examples
banner("Section 13: Confusion matrices + examples")
def save_confusion(y_pred, y_true, classes, title, filename, figsize=(5, 4)):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    save_fig(filename)

# best text
best_text_name = max(text_histories, key=lambda k: text_histories[k].test_acc[-1])
print(f"Best text model: {best_text_name} ({text_histories[best_text_name].test_acc[-1]:.4f})")
best_text_model = {"Transformer": transformer_text, "MLP": mlp_text,
                   "CNN1D": cnn_text, "RNN": rnn_text}[best_text_name]
y_pred_t, y_true_t = collect_predictions(best_text_model, test_loader, device)
save_confusion(y_pred_t.numpy(), y_true_t.numpy(),
               classes=["negative", "positive"],
               title=f"Confusion matrix — {best_text_name} (IMDb)",
               filename="confusion_text.png")

# best vision
best_vision_name = max(vision_histories, key=lambda k: vision_histories[k].test_acc[-1])
print(f"Best vision model: {best_vision_name} ({vision_histories[best_vision_name].test_acc[-1]:.4f})")
best_vision_model = {"ViT": vit, "ImageMLP": image_mlp,
                     "SimpleCNN": simple_cnn, "SmallResNet": small_resnet}[best_vision_name]
y_pred_v, y_true_v = collect_predictions(best_vision_model, vision_test_loader, device)
save_confusion(y_pred_v.numpy(), y_true_v.numpy(),
               classes=CIFAR_CLASSES,
               title=f"Confusion matrix — {best_vision_name} (CIFAR-10)",
               filename="confusion_vision.png",
               figsize=(7.5, 6))

results["best_text_model"] = best_text_name
results["best_vision_model"] = best_vision_name

# Text correct/wrong examples → text file
correct_mask = (y_pred_t == y_true_t).numpy()
wrong_mask   = ~correct_mask
labels_str = ["negative", "positive"]
with (PLOTS / "text_examples.txt").open("w") as f:
    f.write("=== 3 PREDIKSI BENAR ===\n")
    for i in np.where(correct_mask)[0][:3]:
        f.write(f"\n[true={labels_str[y_true_t[i]]}, pred={labels_str[y_pred_t[i]]}]\n")
        f.write(raw_test_texts[i][:400].replace("\n", " ") + "...\n")
    f.write("\n\n=== 3 PREDIKSI SALAH ===\n")
    for i in np.where(wrong_mask)[0][:3]:
        f.write(f"\n[true={labels_str[y_true_t[i]]}, pred={labels_str[y_pred_t[i]]}]\n")
        f.write(raw_test_texts[i][:400].replace("\n", " ") + "...\n")

# Vision correct/wrong grid
viz_test = CIFAR10(root=str(ROOT / "data"), train=False, download=False, transform=T.ToTensor())
correct_idx_v = np.where((y_pred_v == y_true_v).numpy())[0][:4]
wrong_idx_v   = np.where((y_pred_v != y_true_v).numpy())[0][:4]
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for col, idx in enumerate(correct_idx_v):
    img, _ = viz_test[idx]
    axes[0, col].imshow(img.permute(1, 2, 0).numpy())
    axes[0, col].set_title(f"✓ T:{CIFAR_CLASSES[y_true_v[idx]]}\nP:{CIFAR_CLASSES[y_pred_v[idx]]}", fontsize=9)
    axes[0, col].axis("off")
for col, idx in enumerate(wrong_idx_v):
    img, _ = viz_test[idx]
    axes[1, col].imshow(img.permute(1, 2, 0).numpy())
    axes[1, col].set_title(f"✗ T:{CIFAR_CLASSES[y_true_v[idx]]}\nP:{CIFAR_CLASSES[y_pred_v[idx]]}", fontsize=9)
    axes[1, col].axis("off")
plt.suptitle(f"{best_vision_name} — atas: benar, bawah: salah")
save_fig("vision_examples.png")

# ──────────────────────────────────────────────────────────────── 13.3 summary tables
banner("Section 13.3: Summary tables → CSV")
rows = []
def row(name, dataset, h):
    return {
        "Model": name, "Dataset": dataset,
        "Accuracy": round(h.test_acc[-1], 4),
        "Loss": round(h.test_loss[-1], 4),
        "Parameter": h.num_params,
        "Train Time (s)": round(h.train_time, 1),
        "Inference/batch (ms)": round(h.inference_time_ms, 2),
    }
for n, h in text_histories.items():
    rows.append(row(n, "IMDb", h))
for n, h in vision_histories.items():
    rows.append(row(n, "CIFAR-10", h))
summary_df = pd.DataFrame(rows)
summary_df.to_csv(TABLES / "summary.csv", index=False)
print(summary_df.to_string(index=False))

pd.DataFrame(results["seq_len"]).T.to_csv(TABLES / "seq_len_variation.csv")
pd.DataFrame(results["patch_size"]).T.to_csv(TABLES / "patch_size_variation.csv")

# ──────────────────────────────────────────────────────────────── 14-15 Swin2SR
banner("Section 14-15: Swin2SR super-resolution + tiled modification")
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

SR_MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"
print(f"Loading {SR_MODEL_ID} …")
sr_processor = AutoImageProcessor.from_pretrained(SR_MODEL_ID, use_fast=True)
sr_model = Swin2SRForImageSuperResolution.from_pretrained(SR_MODEL_ID).to(device).eval()
sr_params = sum(p.numel() for p in sr_model.parameters())
print(f"Swin2SR params: {sr_params:,}")


def swin2sr_inference(pil_img: Image.Image, model, processor, scale: int = 2) -> Image.Image:
    inputs = processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    sr = out.reconstruction.squeeze(0).clamp(0, 1).cpu().numpy()
    sr = (sr.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    W, H = pil_img.size
    sr = sr[: H * scale, : W * scale]
    return Image.fromarray(sr)


def show_sr_comparison(lr_pil, sr_pil, title, filename):
    bicubic = lr_pil.resize(sr_pil.size, Image.BICUBIC)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(np.asarray(lr_pil));  axes[0].set_title(f"LR original\n{lr_pil.size}");     axes[0].axis("off")
    axes[1].imshow(np.asarray(bicubic)); axes[1].set_title(f"Bicubic upscale\n{bicubic.size}"); axes[1].axis("off")
    axes[2].imshow(np.asarray(sr_pil));  axes[2].set_title(f"Swin2SR ×2\n{sr_pil.size}");      axes[2].axis("off")
    plt.suptitle(title)
    save_fig(filename)


# 2 CIFAR demos
for n_demo, idx in enumerate([3, 7]):
    img_tensor, label = viz_test[idx]
    lr_pil = Image.fromarray((img_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8))
    sr_pil = swin2sr_inference(lr_pil, sr_model, sr_processor)
    show_sr_comparison(lr_pil, sr_pil,
                       f"CIFAR-10 — class={CIFAR_CLASSES[label]}",
                       f"sr_cifar_{n_demo}.png")

# Astronaut demo
try:
    from skimage import data as sk_data
    img_np = sk_data.astronaut()
    medium_pil = Image.fromarray(img_np).resize((128, 128))
except Exception:
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(0, 255, 128, dtype=np.uint8)
    base[:, :, 1] = np.linspace(0, 255, 128, dtype=np.uint8).reshape(-1, 1)
    base[:, :, 2] = 128
    medium_pil = Image.fromarray(base)

sr_medium = swin2sr_inference(medium_pil, sr_model, sr_processor)
show_sr_comparison(medium_pil, sr_medium, "Sample 128×128 → 256×256", "sr_medium.png")


def tiled_super_resolution(pil_img: Image.Image, model, processor,
                           tile: int = 64, overlap: int = 8, scale: int = 2) -> Image.Image:
    """Modifikasi: split big images into overlapping tiles → SR each → blend with feather window."""
    W, H = pil_img.size
    if W <= tile and H <= tile:
        return swin2sr_inference(pil_img, model, processor)
    stride = tile - overlap
    out_W, out_H = W * scale, H * scale
    canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
    weight = np.zeros((out_H, out_W, 1), dtype=np.float32)
    ramp = np.linspace(0, 1, tile * scale, dtype=np.float32)
    win = np.minimum(ramp, ramp[::-1])
    win2d = (win[:, None] * win[None, :])[..., None]
    win2d = np.maximum(win2d, 1e-3)
    img_np = np.asarray(pil_img)
    y = 0
    while y < H:
        y_end = min(y + tile, H)
        y0 = y_end - tile if y_end - y < tile else y
        x = 0
        while x < W:
            x_end = min(x + tile, W)
            x0 = x_end - tile if x_end - x < tile else x
            patch = img_np[y0:y0 + tile, x0:x0 + tile]
            sr_patch = np.asarray(swin2sr_inference(Image.fromarray(patch), model, processor)).astype(np.float32)
            exp = tile * scale
            sr_patch = sr_patch[:exp, :exp]
            ty, tx = y0 * scale, x0 * scale
            canvas[ty:ty + exp, tx:tx + exp] += sr_patch * win2d
            weight[ty:ty + exp, tx:tx + exp] += win2d
            if x_end >= W: break
            x += stride
        if y_end >= H: break
        y += stride
    blended = (canvas / weight).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# Sanity ≤ tile
small_pil = medium_pil.resize((48, 48))
plain = swin2sr_inference(small_pil, sr_model, sr_processor)
tiled = tiled_super_resolution(small_pil, sr_model, sr_processor)
diff_small = float(np.abs(np.asarray(plain).astype(int) - np.asarray(tiled).astype(int)).mean())
print(f"Sanity check (≤ tile): diff = {diff_small:.3f}")

# 128×128 tiled
t0 = time.perf_counter()
tiled_big = tiled_super_resolution(medium_pil, sr_model, sr_processor)
tile_time = time.perf_counter() - t0
plain_big = swin2sr_inference(medium_pil, sr_model, sr_processor)
diff_big = float(np.abs(np.asarray(plain_big).astype(int) - np.asarray(tiled_big).astype(int)).mean())
print(f"Tiled SR on 128×128: time={tile_time:.2f}s, output={tiled_big.size}, diff vs plain={diff_big:.3f}")

# Compare plain vs tiled visually
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
axes[0].imshow(np.asarray(medium_pil));  axes[0].set_title(f"Input {medium_pil.size}");   axes[0].axis("off")
axes[1].imshow(np.asarray(plain_big));   axes[1].set_title(f"Plain SR {plain_big.size}"); axes[1].axis("off")
axes[2].imshow(np.asarray(tiled_big));   axes[2].set_title(f"Tiled SR {tiled_big.size}"); axes[2].axis("off")
plt.suptitle("Modifikasi: Tiled Super-Resolution")
save_fig("sr_tiled_comparison.png")

results["sr"] = {
    "model_id": SR_MODEL_ID,
    "model_params": int(sr_params),
    "sanity_diff_small": diff_small,
    "tiled_time_128": round(tile_time, 2),
    "plain_vs_tiled_diff": round(diff_big, 3),
}

# ──────────────────────────────────────────────────────────────── final save
banner("Saving results.json")
results_path = OUT / "results.json"
with results_path.open("w") as f:
    json.dump(results, f, indent=2)
print(f"All results → {results_path}")
print(f"Plots      → {PLOTS}/ ({len(list(PLOTS.glob('*')))} files)")
print(f"Tables     → {TABLES}/ ({len(list(TABLES.glob('*')))} files)")

print("\n" + "=" * 70)
print("  DONE.")
print("=" * 70)
