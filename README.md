# Panduan Praktikum IF5100

Referensi cepat untuk praktikum **IF5100 Pemrograman untuk Data Analitik** (ITB), Week 11–12.

- **Visualisasi**: import data di Colab (mount Drive atau upload), memilih grafik berdasarkan taksonomi
  (categorical, hierarchical, temporal, relational), snippet matplotlib/seaborn/plotly dengan ilustrasi
  grafik, error umum, cara membaca grafik, dan prinsip desain.
- **Cleaning Data**: diagnosis kualitas data, menyeragamkan teks, placeholder & tipe data, duplikat,
  missing values, outlier (IQR / z-score), buang kolom, encoding & scaling, validasi, dan template
  penjelasan keputusan cleaning.

Tekan `/` untuk mencari snippet. Tombol **Salin** menyalin kode.

## Struktur project

```
index.html            halaman (dibuat oleh tools/build_page.py)
src/main.js           tab, pencarian, tombol salin, highlight kode
src/style.css         tampilan (tema terang & gelap)
public/favicon.svg
tools/build_page.py   isi konten: section(...) dan snip(...)
tools/make_figs.py    menggambar ilustrasi grafik -> tools/figs.json
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
