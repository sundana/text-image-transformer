# Laporan Tugas Transformer: Text Classification, Vision Transformer, dan Image Super-Resolution

## 1. Identitas

| | |
|---|---|
| **Mata Kuliah** | Deep Learning |
| **Nama** | _Sundana Firmansyah_ |
| **NIM** | _[isi NIM]_ |
| **Tanggal** | 12 Mei 2026 |
| **Lingkungan** | Python 3.14, PyTorch 2.11.0+cu130, RTX 5060 Ti (17.1 GB VRAM) |

## 2. Dataset

| Domain | Dataset | Ukuran | Split |
|---|---|---|---|
| Text | **IMDb 50K Movie Reviews** (Kaggle) — review film bahasa Inggris, label sentiment biner | 50.000 review, 25k positif / 25k negatif | 40k train / 10k test (stratified, seed=42) |
| Vision | **CIFAR-10** (torchvision) — 10 kelas objek 32×32 RGB | 60.000 gambar (50k train / 10k test) | bawaan torchvision |
| Super-Resolution | Sample image `skimage.data.astronaut` (downscaled 128×128) + 2 sampel CIFAR-10 (32×32) | inference-only | — |

## 3. Model yang Diimplementasikan

Seluruh blok Transformer & ViT ditulis **dari nol** menggunakan PyTorch (`models/transformer.py`, `models/vit.py`) — tidak memakai `nn.TransformerEncoder` bawaan.

### Text (`max_len=256`)
1. **TransformerClassifier** — embedding + sinusoidal positional encoding + 2× TransformerEncoderBlock (MultiHeadAttention + FFN + LayerNorm) + mean pooling + linear classifier. `d_model=128, heads=4, layers=2, d_ff=256`.
2. **MLPClassifier** — embedding + mean pool + 2 dense layer (256, 128).
3. **CNN1DClassifier (TextCNN)** — embedding + parallel Conv1d kernel sizes [2, 3, 4] + max-over-time pool.
4. **RNNClassifier** — embedding + 2-layer bidirectional RNN + last hidden state.

### Vision (CIFAR-10 32×32)
1. **ViT** — PatchEmbedding (Conv2d 4×4 stride 4) + CLS token + learnable positional embedding + 6× TransformerEncoderBlock (dipakai ulang dari `transformer.py`) + LayerNorm + linear head. `d_model=192, heads=6, layers=6, d_ff=384`.
2. **ImageMLP** — flatten 3·32·32 + FC[512, 256] + dropout.
3. **SimpleCNN** — 3 conv block (BN + ReLU + MaxPool) + GAP.
4. **SmallResNet** — 3 stage ResNet (32→64→128) dengan 6 BasicBlock, tanpa torchvision.

Setiap model dilatih dengan setting yang sama untuk perbandingan adil:
> Adam (lr=1e-3, weight_decay=1e-5), CrossEntropyLoss, 10 epoch, batch 64 (text) / 128 (vision). Untuk vision: augmentasi `RandomCrop(32, padding=4) + RandomHorizontalFlip`.

## 4. Hasil Eksperimen

### Tabel 1 — Hasil Text & Vision (10 epoch)

| Model | Dataset | Accuracy | Loss | Parameter | Train Time | Inference/batch | Catatan |
|---|---|---|---|---|---|---|---|
| **Transformer** | IMDb | **0.8751** | 0.4827 | 2.84 M | 120.7 s | 5.68 ms | terbaik di text |
| MLP | IMDb | 0.8703 | 0.5968 | 2.63 M | 13.9 s | 0.46 ms | tercepat |
| CNN1D | IMDb | 0.8682 | 0.6280 | 2.71 M | 23.6 s | 0.83 ms | overfit cepat |
| RNN | IMDb | 0.7723 | 0.5342 | 2.73 M | 26.8 s | 0.84 ms | tidak stabil |
| ViT | CIFAR-10 | 0.6603 | 0.9684 | 1.81 M | 134.1 s | 10.58 ms | — |
| ImageMLP | CIFAR-10 | 0.4003 | 1.6271 | 1.71 M | 60.7 s | 8.20 ms | terlemah |
| SimpleCNN | CIFAR-10 | 0.6151 | 1.1253 | **0.095 M** | 69.4 s | 9.05 ms | paling efisien |
| **SmallResNet** | CIFAR-10 | **0.7617** | 0.7594 | 0.70 M | 114.7 s | 8.62 ms | **terbaik di vision** |

Visualisasi: `output/plots/text_training_curves.png`, `output/plots/vision_training_curves.png`.
Confusion matrix model terbaik: `output/plots/confusion_text.png`, `output/plots/confusion_vision.png`.
Contoh prediksi benar/salah: `output/plots/text_examples.txt`, `output/plots/vision_examples.png`.

### Tabel 2 — Variasi Sequence Length (Transformer @ IMDb, 5 epoch)

| max_len | Test accuracy | Train time | Catatan |
|---|---|---|---|
| 64 | 0.8078 | 18.8 s | banyak konteks terpotong |
| 128 | 0.8469 | 21.5 s | mulai cukup |
| **256** | **0.8877** | 60.3 s | sweet spot (median review 173 kata) |
| 512 | 0.8932 | 192.4 s | gain marginal, compute 3× |

Plot: `output/plots/variations.png` (kiri).

### Tabel 3 — Variasi Patch Size (ViT @ CIFAR-10, 5 epoch)

| patch_size | # patches | Test accuracy | Train time | Catatan |
|---|---|---|---|---|
| 2 | 256 | **0.5770** | 335.4 s | terbaik, tapi 5× lebih lambat |
| 4 | 64 | 0.5444 | 66.9 s | sweet spot praktis |
| 8 | 16 | 0.3671 | 48.0 s | konteks spasial hilang |
| 16 | 4 | 0.3113 | 48.2 s | praktis rusak |

Plot: `output/plots/variations.png` (kanan).

## 5. Implementasi GitHub Non-Klasifikasi — Swin2SR Super-Resolution

### Tabel 4 — Detail Repository

| Aspek | Isi |
|---|---|
| **Link Repository** | https://github.com/mv-lab/swin2sr |
| **Paper** | Conde et al., "Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration", ECCV-W 2022 |
| **Model pretrained** | `caidas/swin2SR-classical-sr-x2-64` (Hugging Face) — 12.09 M parameter |
| **Jenis Tugas** | Single Image Super-Resolution (SISR) — *bukan klasifikasi* |
| **Input** | Gambar low-resolution (PIL Image) |
| **Output Model** | Gambar high-resolution dengan skala 2× |
| **Dataset/Input** | 2 sampel CIFAR-10 32×32 → 64×64 + foto astronaut 128×128 → 256×256 |
| **Mode** | Inference-only (sesuai catatan CLAUDE.md untuk repo berat) |

**Mengapa bukan klasifikasi?** Output Swin2SR adalah tensor RGB beresolusi penuh — model menghasilkan **gambar baru**, bukan probabilitas kelas. Tugas ini termasuk kategori *image restoration*.

Hasil visual: `output/plots/sr_cifar_0.png`, `sr_cifar_1.png`, `sr_medium.png`.

## 6. Modifikasi Kode: Tiled Super-Resolution

### Masalah pada implementasi standar
Swin2SR dilatih pada window 64×64. Pada gambar besar:
1. Kompleksitas attention O(N²) per window — VRAM melonjak.
2. GPU memori terbatas → potensi Out-Of-Memory.

### Solusi yang diimplementasikan
Ditambahkan fungsi baru `tiled_super_resolution(model, processor, pil_img, tile=64, overlap=8)` di `main.py` (juga ada di notebook). Algoritma:
1. Gambar besar dipecah menjadi tile **64×64** dengan **overlap 8 piksel**.
2. Setiap tile di-SR independen.
3. Output tile (128×128) digabung kembali ke kanvas dengan **feathered blending** (triangular window) → mencegah seam visible.
4. Untuk gambar `W×H ≤ tile`, fungsi falls back ke inference biasa (identik).

### Validasi
| Skenario | Diff vs plain inference | Catatan |
|---|---|---|
| 48×48 (≤ tile) | **0.000** | identik — verified |
| 128×128 (> tile) | 0.139 / 255 ≈ 0.05% | virtually identical |
| Waktu 128×128 | 0.39 s | 4 tiles × ~0.1 s |

Perbandingan visual: `output/plots/sr_tiled_comparison.png`.

**Pengaruh modifikasi:**
- ✅ Memungkinkan SR pada gambar berukuran arbitrer tanpa OOM.
- ✅ Output identik (mean abs diff < 1 pixel) dengan inference biasa.
- ⚖️ Trade-off: sedikit lebih lambat karena beberapa forward pass per gambar.

## 7. Analisis Hasil (Jawaban 11 Pertanyaan CLAUDE.md)

**1. Model mana yang paling akurat?**
- Text: **Transformer** (87.51%) tipis di atas MLP (87.03%) dan CNN1D (86.82%). RNN tertinggal jauh (77.23%).
- Vision: **SmallResNet** (76.17%) unggul mutlak, ViT (66.03%) di posisi kedua.

**2. Model mana yang paling cepat?**
- Text training: **MLP** 13.9 s (paling cepat) vs Transformer 120.7 s (9× lebih lambat).
- Text inference: MLP 0.46 ms/batch — Transformer 5.68 ms/batch (12× lebih lambat).
- Vision: ImageMLP/SimpleCNN ~60-70 s vs ViT 134 s. SimpleCNN paling efisien (94k param, 0.62 acc).

**3. Apakah Transformer selalu lebih baik?**
**Tidak.** Di IMDb gainnya hanya 0.5pp dibanding MLP biasa. Di CIFAR-10, ViT kalah 9.5pp dari SmallResNet. Konsisten dengan temuan paper ViT: tanpa pretraining besar, Transformer tidak unggul di dataset kecil.

**4. Mengapa CNN masih kuat untuk gambar kecil?**
CNN punya **inductive bias** yang sesuai untuk gambar: translation equivariance + locality. Pada CIFAR-10 yang hanya punya 50k training samples, prior tersebut sangat membantu. ViT harus *mempelajari* hal yang sama dari data — butuh data jauh lebih banyak. SmallResNet bahkan dengan ~3× lebih sedikit parameter (697k vs 1.81M) mengalahkan ViT sebesar 9.5pp.

**5. Mengapa LSTM/CNN 1D masih relevan untuk teks pendek?**
IMDb didominasi kata-kata kunci lokal ("amazing", "boring", "great"). CNN1D dengan n-gram filter (kernel 2-4) menangkap pola ini efisien. Hasil: CNN1D 86.82% — hanya 0.7pp di bawah Transformer dengan 5× lebih sedikit waktu training. Untuk teks pendek tanpa long-range dependency, attention overkill.

**6. Apa kelemahan Transformer pada dataset kecil?**
- Banyak parameter, sedikit prior → cepat overfit. Lihat Transformer-IMDb: train acc 97.96% sementara test 87.51% (gap 10pp).
- Variance tinggi: butuh dropout, weight decay, augmentasi.
- Pretraining adalah obat terbaik — tidak tersedia di setup ini, sehingga ViT under-perform.

**7. Apa pengaruh patch size pada ViT?**
- `patch=2` (256 patches): terbaik (57.7%) tapi 5× lebih lambat — banyak token, kompleksitas attention tinggi.
- `patch=4` (64 patches): sweet spot praktis (54.4%, 67 s).
- `patch=8` (16 patches): konteks spasial hilang → 36.7%.
- `patch=16` (4 patches): praktis rusak (31.1%) — 1 patch ≈ 1/4 gambar.

Rule of thumb: **lebih banyak patch = lebih halus, tetapi lebih mahal** secara kuadrat.

**8. Apa pengaruh sequence length pada teks?**
- `len=64`: terlalu pendek (80.8%).
- `len=128`: 84.7%.
- `len=256`: 88.8% — terbaik per unit waktu.
- `len=512`: 89.3% — gain 0.5pp tapi compute 3× → diminishing returns. Median review IMDb 173 kata, jadi >256 hanya menambah padding.

**9. Apa input dan output repository GitHub non-klasifikasi?**
Swin2SR: **Input** = gambar low-resolution (PIL Image). **Output** = gambar high-resolution skala 2× (PIL Image). Bukan label kelas.

**10. Mengapa tugas GitHub tersebut bukan klasifikasi?**
Output Swin2SR adalah **tensor RGB beresolusi penuh** — model menghasilkan piksel-piksel baru pada resolusi 2× lebih besar. Ini termasuk kategori *image restoration* / *image-to-image translation*, bukan kategorisasi diskrit.

**11. Modifikasi apa yang dilakukan dan bagaimana pengaruhnya?**
Implementasi `tiled_super_resolution()` (tile 64×64, overlap 8 px, feathered blending). Pengaruh:
- ✅ Mendukung gambar berukuran arbitrer tanpa OOM.
- ✅ Output identik dengan plain inference (mean abs diff 0.139/255 ≈ 0.05%).
- ✅ Sanity check pada gambar ≤ tile menghasilkan diff = 0.000 (identik).
- ⚖️ Sedikit lebih lambat (multiple forward pass).

## 8. Kesimpulan

1. **Transformer tidak selalu lebih baik.** Pada dataset kecil & tugas klasifikasi sederhana, model dengan inductive bias yang sesuai (CNN, ResNet, CNN1D) sering mengungguli Transformer baik dalam akurasi (CIFAR-10: SmallResNet 76% vs ViT 66%) maupun efisiensi (MLP 14 s vs Transformer 121 s untuk IMDb).

2. **Inductive bias matters.** CNN locality + translation equivariance sangat penting di CIFAR-10. SmallResNet dengan **3× lebih sedikit** parameter mengalahkan ViT sebesar 9.5pp.

3. **Hyperparameter struktural (patch size, sequence length) berpengaruh besar.** Patch terlalu besar atau seq terlalu pendek membuat model rusak. Sweet spot tergantung karakteristik dataset.

4. **Transformer rentan overfitting** pada dataset kecil — perlu regularisasi kuat atau pretraining masif (JFT-300M di paper ViT asli). Setup from-scratch dengan 50k sampel tidak cukup.

5. **Untuk tugas non-klasifikasi** (super-resolution), Transformer-based (Swin) memang state-of-the-art. Modifikasi sederhana seperti **tiled inference** membuat model lebih praktis untuk gambar berukuran arbitrer.

6. **Tujuan tugas tercapai**: memahami implementasi Transformer dari nol, membandingkan trade-off antar arsitektur, menjalankan + memodifikasi repository nyata, dan menginterpretasi hasil eksperimen.

## 9. Output yang Dihasilkan

```
output/
├── plots/
│   ├── imdb_length_distribution.png       (distribusi panjang review)
│   ├── cifar10_samples.png                (sampel per kelas CIFAR-10)
│   ├── text_training_curves.png           (loss + accuracy 4 model text)
│   ├── vision_training_curves.png         (loss + accuracy 4 model vision)
│   ├── variations.png                     (bar chart seq_len & patch_size)
│   ├── confusion_text.png                 (CM model text terbaik)
│   ├── confusion_vision.png               (CM model vision terbaik)
│   ├── text_examples.txt                  (3 prediksi benar + 3 salah text)
│   ├── vision_examples.png                (grid prediksi benar/salah vision)
│   ├── sr_cifar_0.png, sr_cifar_1.png     (Swin2SR pada CIFAR-10)
│   ├── sr_medium.png                      (Swin2SR pada astronaut 128×128)
│   └── sr_tiled_comparison.png            (plain SR vs tiled SR)
├── tables/
│   ├── summary.csv                        (Tabel 1)
│   ├── seq_len_variation.csv              (Tabel 2)
│   └── patch_size_variation.csv           (Tabel 3)
└── results.json                           (semua metrik dalam JSON)
```

Kode utama:
- `main.py` — script end-to-end (run: `conda activate deep-learning && python main.py`)
- `transformer_assignment.ipynb` — versi notebook 16 section
- `models/{transformer,vit,mlp,cnn1d,rnn,vision_baselines}.py`
- `utils/{text_data,training}.py`

## 10. Referensi

1. Vaswani, A., et al. *"Attention Is All You Need."* NeurIPS 2017.
2. Dosovitskiy, A., et al. *"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale."* ICLR 2021.
3. Kim, Y. *"Convolutional Neural Networks for Sentence Classification."* EMNLP 2014.
4. He, K., et al. *"Deep Residual Learning for Image Recognition."* CVPR 2016.
5. Conde, M., et al. *"Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration."* ECCV Workshops 2022. **Repository: https://github.com/mv-lab/swin2sr**.
6. Liang, J., et al. *"SwinIR: Image Restoration Using Swin Transformer."* ICCV Workshops 2021.
7. Maas, A., et al. *"Learning Word Vectors for Sentiment Analysis."* ACL 2011 — dataset IMDb.
8. Krizhevsky, A. *"Learning Multiple Layers of Features from Tiny Images."* Technical Report, 2009 — dataset CIFAR-10.
9. Hugging Face Transformers — model card `caidas/swin2SR-classical-sr-x2-64`.
10. PyTorch documentation — https://pytorch.org/docs/stable/.
