# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses the `deep-learning` conda environment:

```bash
conda activate deep-learning
```

Key packages: Python 3.14, PyTorch 2.11, torchvision 0.26, NumPy 2.4.

## Tujuan

Mahasiswa diminta mengimplementasikan Transformer menggunakan Python dan PyTorch pada dua domain:

1. Text Classification
2. Image Classification / Vision Transformer
Selain itu, mahasiswa wajib mencoba 1 repository GitHub untuk tugas non-klasifikasi, misalnya object detection, segmentation, summarization, translation, image captioning, super-resolution, denoising, deblurring, atau image generation.

Fokus tugas ini adalah memahami alur model, menjalankan eksperimen, membandingkan hasil, dan menganalisis kelebihan serta kelemahan Transformer.

## Implementasi Text Transformer

### Dataset

Mahasiswa boleh memilih salah satu:

- IMDb Review Dataset
- AG News Dataset
- SST-2
- Dataset sentimen Bahasa Indonesia
- Dataset buatan sendiri minimal 100 kalimat

### Model yang wajib dibuat:

1. Transformer untuk text classification
2. Model pembanding: MLP, CNN 1D, RNN, LSTM, atau GRU

### Arsitektur minimal Transformer Text:

Input Text
→ Tokenization
→ Embedding
→ Positional Encoding
→ Transformer Encoder
→ Pooling
→ Linear Classifier
→ Output Class

### Komponen PyTorch yang dapat digunakan:

- nn.Embedding
- nn.TransformerEncoderLayer
- nn.TransformerEncoder
- nn.Linear
- nn.CrossEntropyLoss
- torch.optim.Adam
  
### Yang harus dianalisis:

Apakah Transformer lebih baik dari LSTM/CNN/MLP?
Bagaimana pengaruh panjang kalimat?
Apakah Transformer lebih lambat atau lebih berat?
Apakah model pembanding lebih stabil pada dataset kecil?


## Implementasi Vision Transformer

### Dataset

Mahasiswa boleh memilih salah satu:

- MNIST
- Fashion-MNIST
- CIFAR-10
- Dataset gambar sederhana dari Kaggle
- Dataset gambar buatan sendiri

### Model yang wajib dibuat:

1. Vision Transformer sederhana
2. Model pembanding: MLP, CNN sederhana, ResNet kecil, atau MobileNet kecil

### Arsitektur minimal Vision Transformer:

Input Image
→ Patch Splitting
→ Patch Embedding
→ Positional Embedding
→ Transformer Encoder
→ Classification Head
→ Output Class
  
Contoh:

Image 32×32
Patch size 4×4
Jumlah patch = 64
Patch → embedding vector → Transformer Encoder → classifier
  
### Yang harus dianalisis:

Apakah ViT lebih baik dari CNN pada dataset kecil?
Apakah CNN lebih cepat dilatih?
Apa pengaruh ukuran patch?
Apakah ViT mudah overfitting?


## Implementasi GitHub Non-Klasifikasi

Mahasiswa wajib mencari dan menjalankan minimal 1 repository GitHub untuk tugas selain klasifikasi.

### Topik yang boleh dipilih:

Object Detection
Image Segmentation / Semantic Segmentation
Image Captioning
Machine Translation
Text Summarization
Question Answering
Super-Resolution
Denoising / Deblurring
Depth Estimation
Image Generation

Contoh model/repository yang boleh dicari:

DETR / YOLO untuk object detection
U-Net / SegFormer untuk segmentation
T5 / BART untuk summarization atau translation
BLIP untuk image captioning
SwinIR / ESRGAN untuk super-resolution
Diffusion model sederhana untuk image generation

### Yang harus dilakukan:

Tulis link repository GitHub.
Jelaskan jenis tugas dan model yang digunakan.
Jalankan training, testing, atau inference.
Tampilkan hasil output model.
Modifikasi minimal satu bagian kode.
Jelaskan hasil sebelum dan sesudah modifikasi.

Catatan: Repository GitHub tambahan tidak boleh hanya berisi tugas klasifikasi.


## Eksperimen wajib

| Bagian | Eksperimen | Tujuan |
|---|---|---|
| Text  |	Transformer vs LSTM/CNN/MLP     |	Membandingkan performa model teks |
| Vision    |	Vision Transformer vs CNN/MLP/ResNet    |	Membandingkan performa model gambar |
| Vision |	Variasi patch size pada ViT |	Melihat pengaruh ukuran patch |
| Text |	Variasi panjang sequence |	Melihat pengaruh panjang input teks |
| GitHub |	1 repository non-klasifikasi |	Memahami implementasi proyek nyata |

## Visualisasi dan Evaluasi

Mahasiswa wajib menampilkan:

1. Training loss
2. Training / testing accuracy
3. Confusion matrix
4. Contoh prediksi benar dan salah
5. Jumlah parameter model
6. Waktu training atau inference

Untuk GitHub non-klasifikasi, tampilkan output sesuai tugasnya, misalnya bounding box, mask segmentasi, hasil ringkasan teks, hasil terjemahan, hasil super-resolution, atau gambar hasil generation.


## Tabel yang wajib dibuat 

### 1. Tabel Text dan Vision

| Model |	Dataset	Accuracy |	Loss |	Parameter | 	Waktu |	Catatan |
|---|---|---|---|---|---|
| Transformer Text |	... |	... |	... |	... |	... |	... |
| LSTM / CNN 1D / MLP |	... |	... |	... |	... |	... |	... |
| Vision Transformer |	... |	... |	... |	... |	... |	... |
| CNN / MLP / ResNet |	... |	... |	... |	... |	... |	... |



### 2. Tabel Github Non-klasifikasi

| Aspek |	Isi |
| --- | --- |
| Link Repository	...
| Jenis Tugas	Detection / Segmentation / Translation / Summarization / Restoration / Generation |
| Model |	... |
| Dataset/Input	 |... |
| Output Model |	... |
| Modifikasi Kode	 |... |
| Hasil Setelah Modifikasi |	... |


## Analisis yang Wajib Ditulis
1. Model mana yang paling akurat?
2. Model mana yang paling cepat?
3. Apakah Transformer selalu lebih baik?
4. Mengapa CNN masih kuat untuk gambar kecil?
5. Mengapa LSTM/CNN 1D masih relevan untuk teks pendek?
6. Apa kelemahan Transformer pada dataset kecil?
7. Apa pengaruh patch size pada ViT?
8. Apa pengaruh sequence length pada teks?
9. Apa input dan output repository GitHub non-klasifikasi?
10. Mengapa tugas GitHub tersebut bukan klasifikasi?
11. Modifikasi apa yang dilakukan dan bagaimana pengaruhnya?

## Struktur Notebook
1. Import Library
2. Set Device
3. Load Dataset Text
4. Preprocessing Text
5. Model Transformer Text
6. Model Pembanding Text
7. Training dan Evaluation Text
8. Load Dataset Vision
9. Model Vision Transformer
10. Model Pembanding Vision
11. Training dan Evaluation Vision
12. Plot Loss dan Accuracy
13. Confusion Matrix
14. Implementasi GitHub Non-Klasifikasi
15. Modifikasi Kode GitHub
16. Analisis dan Kesimpulan

## Ketentuan Teknis
1. Wajib menggunakan Python dan PyTorch untuk implementasi utama.
2. Minimal training 5 epoch untuk implementasi sendiri.
3. Boleh menggunakan Google Colab.
4. Kode harus diberi komentar pada bagian penting.
5. Jika menggunakan kode GitHub, wajib mencantumkan sumber.
6. GitHub tambahan harus berupa tugas non-klasifikasi.
7. Jika training GitHub terlalu berat, boleh menjalankan inference/demo.


Cek device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```
  
Hitung jumlah parameter:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```
  
Hitung waktu training/inference:

```python
import time

start = time.time()

# training atau inference

end = time.time()
print("Running time:", end - start, "seconds")
```


## Output yang Dikumpulkan
1. Notebook .ipynb
2. File .py, jika ada
3. Link dataset
4. Link repository GitHub non-klasifikasi
5. Grafik loss dan accuracy
6. Confusion matrix
7. Tabel hasil eksperimen
8. Screenshot output GitHub non-klasifikasi
9. Laporan singkat 3–5 halaman
10. Link GitHub atau Google Drive pengumpulan


## Format Laporan Singkat
1. Judul
2. Nama dan NIM
3. Dataset yang digunakan
4. Model yang diimplementasikan
5. Hasil eksperimen
6. Implementasi GitHub non-klasifikasi
7. Modifikasi kode
8. Analisis hasil
9. Kesimpulan
10. Referensi / sumber kode


## Catatan Penting
Tujuan tugas ini bukan mencari akurasi tertinggi, tetapi memahami implementasi, komparasi, dan cara membaca kode proyek nyata.

Untuk teks:
kalimat → token → embedding → positional encoding → self-attention → klasifikasi

Untuk gambar:
gambar → patch → patch embedding → positional encoding → self-attention → klasifikasi

Untuk GitHub non-klasifikasi:
repository → pahami kode → jalankan → modifikasi → tampilkan output → analisis

Larangan:

Tidak boleh hanya copy-paste kode tanpa memahami.
Tidak boleh memilih repository GitHub tambahan yang hanya klasifikasi.
Tidak boleh mengumpulkan tanpa hasil running.
Tidak boleh hanya menampilkan angka tanpa analisis.