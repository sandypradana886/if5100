# Panduan Praktikum IF5100

Referensi cepat untuk praktikum **IF5100 Pemrograman untuk Data Analitik** (ITB), Week 11–12.

- **Visualisasi**: import data di Colab (mount Drive atau upload), memilih grafik berdasarkan taksonomi
  (categorical, hierarchical, temporal, relational), **bentuk data tiap grafik** (mentah atau perlu
  `groupby` / `crosstab` / `pivot` / `melt`), snippet matplotlib/seaborn/plotly dengan ilustrasi grafik,
  error umum, cara membaca grafik, dan prinsip desain.
- **Cleaning Data**: diagnosis kualitas data, menyeragamkan teks, placeholder & tipe data, duplikat,
  missing values, outlier (IQR / z-score), buang kolom, encoding & scaling, validasi, dan template
  penjelasan keputusan cleaning.
- **Prediksi Soal**: studi kasus *Kopi Senja* dengan 7 soal kode (cleaning, 5 jenis visualisasi, debugging),
  5 tugas analisis, dan 7 soal teori. Setiap jawaban sudah dijalankan; grafik dan angka di kunci adalah output
  aslinya. Notebook soal & kunci bisa diunduh dari halaman (`public/notebook/`).
- **Materi**: daftar slide kuliah yang menjadi dasar isi web, link dokumentasi pandas/matplotlib/seaborn/plotly,
  Colab I/O, 6 dataset Kaggle dari slide 12b, alat bantu memilih grafik & warna, dan buku rujukan.

Tekan `/` untuk mencari snippet. Tombol **Salin** menyalin kode.

## Struktur project

```
index.html            halaman (dibuat oleh tools/build_page.py)
src/main.js           tab, pencarian, tombol salin, highlight kode
src/style.css         tampilan (tema terang & gelap)
public/favicon.svg
tools/build_page.py   isi konten: section(...) dan snip(...)
tools/make_figs.py    menggambar ilustrasi grafik -> tools/figs.json
tools/make_prediksi.py soal prediksi: menjalankan jawaban -> tools/prediksi.json + public/notebook/*.ipynb
vercel.json           konfigurasi deploy Vercel
```

## Menjalankan di komputer

Butuh Node.js 22.12 atau lebih baru.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # hasil di folder dist/
npm run preview    # cek hasil build
```

## Mengubah konten

1. Edit `tools/build_page.py` (teks, snippet, tabel).
2. Jalankan `npm run content` (atau `python tools/build_page.py`) untuk membuat ulang `index.html`.
3. Untuk mengubah ilustrasi grafik: `pip install -r tools/requirements.txt`, lalu `python tools/make_figs.py`,
   lalu jalankan langkah 2.
4. Untuk mengubah soal prediksi: edit `tools/make_prediksi.py`, jalankan `python tools/make_prediksi.py`
   (semua jawaban dieksekusi ulang dan angka di penjelasan dihitung otomatis), lalu jalankan langkah 2.

## Deploy ke Vercel

**Lewat dashboard (paling mudah)**

1. Push repo ini ke GitHub.
2. Buka https://vercel.com/new, pilih **Import** pada repo `if5100`.
3. Vercel mendeteksi Vite otomatis (build `npm run build`, output `dist`). Klik **Deploy**.
4. Setiap push ke `main` akan otomatis di-deploy ulang.

**Lewat CLI**

```bash
npm i -g vercel
vercel          # deploy preview
vercel --prod   # deploy production
```

Nama domain bisa diubah di **Project Settings → Domains**, misalnya `if5100.vercel.app` (kalau masih tersedia).

Sumber materi: slide IF5100-11 (pre-processing), IF5100-12a Data Visualization, dan IF5100-12b
Independent Study (Andy Kirk 2012; Iliinsky & Steele 2011; Spoerri).
