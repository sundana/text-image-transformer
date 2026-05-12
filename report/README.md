# Laporan LaTeX

Format makalah akademik (dua-kolom A4, Bahasa Indonesia) untuk tugas Transformer.

## Cara Render ke PDF

### Opsi 1 — TeX Live di Windows (terdeteksi di sistem ini)
```powershell
cd C:\Users\USER\... \college\machine-learning\report
pdflatex main.tex
pdflatex main.tex          # pass kedua untuk resolve cross-references
```

### Opsi 2 — TeX Live di Linux/WSL
```bash
sudo apt install texlive-latex-recommended texlive-fonts-recommended texlive-lang-other
cd report
pdflatex main.tex && pdflatex main.tex
```

### Opsi 3 — Overleaf (online, paling mudah)
1. Buka https://overleaf.com → New Project → Upload Project
2. Upload `main.tex` + folder `figures/` (drag & drop).
3. Klik **Recompile** → download PDF.

### Opsi 4 — VS Code + LaTeX Workshop
Install ekstensi `LaTeX Workshop`, lalu open `main.tex` dan tekan **Ctrl+Alt+B**.

## Struktur

```
report/
├── main.tex            # source LaTeX (dua-kolom, conference-style)
├── figures/            # 12 PNG (training curves, confusion matrices, SR results, dll.)
└── README.md           # file ini
```

## Catatan

- Dokumen menggunakan `article` class + paket `babel[indonesian]`, `geometry`, `multicol`, `booktabs`, `subcaption`, `listings`, `hyperref`.
- Referensi internal menggunakan `\thebibliography` (tidak perlu BibTeX).
- Pass kompilasi cukup dua kali (tidak perlu BibTeX/biber).
- Ganti `NIM: [diisi mahasiswa]` di bagian author dengan NIM Anda.
