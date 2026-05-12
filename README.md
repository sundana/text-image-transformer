# Implementasi dan Analisis Komparatif Transformer

Tugas mata kuliah **Topik Pembelajaran Mesin** — implementasi *from-scratch* Transformer pada tiga domain: klasifikasi teks, klasifikasi gambar (Vision Transformer), dan *super-resolution* gambar non-klasifikasi.

---

## Dataset

### 1. IMDb Large Movie Review Dataset
- **Tugas:** Klasifikasi sentimen biner (positif / negatif)
- **Ukuran:** 50.000 review film berbahasa Inggris (25k train, 25k test), seimbang antar kelas
- **Link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Lokasi lokal:** `data/imdb/IMDB Dataset.csv`

### 2. CIFAR-10
- **Tugas:** Klasifikasi gambar 10 kelas (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Ukuran:** 60.000 gambar 32×32 piksel (50k train, 10k test)
- **Link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Lokasi lokal:** `data/cifar-10-batches-py/`

### 3. Swin2SR — Single Image Super-Resolution (Non-Klasifikasi)
- **Tugas:** *Super-resolution* 2× — menghasilkan gambar beresolusi tinggi dari gambar beresolusi rendah
- **Repository:** https://github.com/mv-lab/swin2sr
- **Checkpoint:** `caidas/swin2SR-classical-sr-x2-64` via Hugging Face
- **Input uji:** gambar CIFAR-10 (32×32 → 64×64)

---

## Struktur Proyek

```
machine-learning/
│
├── transformer_assignment.ipynb   # Notebook utama (16 bagian lengkap)
├── main.py                        # Pipeline end-to-end (mirror notebook)
├── build_notebook.py              # Skrip build notebook dari modul
│
├── models/                        # Definisi arsitektur model
│   ├── __init__.py
│   ├── transformer.py             # TransformerClassifier (teks)
│   ├── mlp.py                     # MLPClassifier (teks) & ImageMLP (gambar)
│   ├── cnn1d.py                   # CNN1D untuk klasifikasi teks
│   ├── rnn.py                     # RNN bidrectional untuk klasifikasi teks
│   ├── vit.py                     # Vision Transformer (ViT)
│   └── vision_baselines.py        # SimpleCNN & SmallResNet
│
├── utils/                         # Utilitas pendukung
│   ├── __init__.py
│   ├── text_data.py               # Tokenisasi & dataset IMDb
│   └── training.py                # Loop training, evaluasi, plotting
│
├── data/                          # Dataset mentah (tidak di-commit)
│   ├── imdb/
│   │   └── IMDB Dataset.csv
│   ├── imdb-dataset-of-50k-movie-reviews.zip
│   ├── cifar-10-python.tar.gz
│   └── cifar-10-batches-py/
│
├── output/                        # Hasil eksperimen (di-generate otomatis)
│   ├── plots/                     # Grafik loss, accuracy, confusion matrix, dll.
│   ├── tables/                    # Tabel CSV hasil variasi hiperparameter
│   │   ├── summary.csv
│   │   ├── seq_len_variation.csv
│   │   └── patch_size_variation.csv
│   └── results.json               # Semua metrik dalam satu file JSON
│
└── report/                        # Laporan PDF (LaTeX)
    ├── main.tex
    ├── main.pdf
    └── figures/                   # Gambar yang digunakan di laporan
```

---

## Cara Menjalankan

### Prasyarat
```bash
conda activate deep-learning
# Python 3.14, PyTorch 2.11, torchvision 0.26, NumPy 2.4
```

### Jalankan pipeline lengkap
```bash
python main.py
```
Semua output (plot, tabel, metrik) disimpan otomatis ke folder `output/`. Estimasi waktu ±22 menit pada GPU NVIDIA RTX 5060 Ti.

### Jalankan via notebook
Buka `transformer_assignment.ipynb` di Jupyter Lab / VS Code dan jalankan sel per bagian.

---

## Ringkasan Hasil

| Model | Dataset | Accuracy | Params | Waktu Train |
|---|---|---|---|---|
| Transformer | IMDb | **87,51%** | 2,84 M | 120,7 s |
| MLP | IMDb | 87,03% | 2,63 M | 13,9 s |
| CNN1D | IMDb | 86,82% | 2,71 M | 23,6 s |
| RNN | IMDb | 77,23% | 2,73 M | 26,8 s |
| ViT | CIFAR-10 | 66,03% | 1,81 M | 134,1 s |
| ImageMLP | CIFAR-10 | 40,03% | 1,71 M | 60,7 s |
| SimpleCNN | CIFAR-10 | 61,51% | 95 k | 69,4 s |
| **SmallResNet** | CIFAR-10 | **76,17%** | 697 k | 114,7 s |
