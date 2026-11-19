"""Content for the NumPy (P06), Pandas (P07) and Scikit-learn (P09) tabs, plus the
"Resep Praktikum 08" section of the Visualisasi tab. Imported by tools/build_page.py.

Each tab: dict(id, label, flow, fixture, sections=[...])
section: dict(id, title, desc, cards=[...]); card: dict(t, code | html, note, wide, test)

Run `python tools/content_p06_p09.py` to execute every code card against its tab fixture
(test=False skips a card). Needs numpy, pandas, matplotlib, seaborn, scipy, scikit-learn."""

NUMPY = dict(
    id="numpy", label="NumPy", count_label="P06",
    placeholder="Cari: axis, linspace, broadcasting, percentile, rng…",
    flow=["Buat array", "Index / mask", "Agregasi per axis", "Operasi vektor", "Statistik", "Simulasi"],
    fixture=r"""
import numpy as np
import random
import matplotlib.pyplot as plt
""",
    sections=[
        dict(id="n-buat", title="Membuat array", desc="Array NumPy: satu tipe data, ukuran tetap, operasi vektor (tanpa loop).", cards=[
            dict(t="Dari list & atribut dasar", code=r"""
a = np.array([1, 2, 3, 4])
M = np.array([[80, 75, 90],
              [60, 85, 70]])          # 2 baris × 3 kolom
M.shape, M.ndim, M.size, M.dtype      # (2, 3), 2, 6, int64
M.astype(float)                       # ubah tipe
np.array([1, 2.5, 'a'])               # campur → semua jadi string!"""),
            dict(t="Array khusus", code=r"""
np.zeros((2, 3)); np.ones(5); np.full((2, 2), 7)
np.eye(3)                      # matriks identitas
np.arange(0, 6, 0.25)          # [0, 6) langkah 0.25 (stop TIDAK ikut)
np.linspace(0, 1, 5)           # 5 titik dari 0 s.d. 1 (stop ikut)
np.empty_like(M)""", note="<b>arange</b> pakai langkah, <b>linspace</b> pakai jumlah titik."),
            dict(t="Reshape, flatten, gabung", code=r"""
b = np.arange(12).reshape(3, 4)   # -1 = hitung otomatis: reshape(-1, 1) → kolom
b.T                               # transpose
b.flatten(); b.ravel()            # jadi 1D (flatten selalu copy)
np.concatenate([a, a])            # 1D
np.vstack([M, M]); np.hstack([M, M])
np.column_stack([a, a ** 2])"""),
            dict(t="Bilangan acak (Generator, cara modern)", code=r"""
rng = np.random.default_rng(42)          # seed → hasil bisa direproduksi
rng.integers(1, 7, size=10)              # dadu: 1..6 (high eksklusif)
rng.uniform(0, 10, size=5)
rng.normal(loc=27, scale=7, size=100)    # mean 27, std 7
rng.exponential(scale=5, size=100)       # waktu tunggu, mean 5
rng.choice(['A', 'B', 'C'], size=10, p=[.5, .3, .2])
rng.permutation(10); rng.shuffle(a)""", note="Gaya lama: <code>np.random.seed(42)</code> + <code>np.random.randint/rand/randn</code>. Modul <code>random</code> bawaan: <code>random.seed</code>, <code>random.random()</code>, <code>random.expovariate(1/mean)</code>, <code>random.choices</code>."),
        ]),
        dict(id="n-index", title="Indexing, slicing & boolean mask", desc="Baris dulu, lalu kolom: a[baris, kolom].", cards=[
            dict(t="Slicing 1D & 2D", code=r"""
a = np.array([10, 20, 30, 40, 50])
a[0], a[-1], a[1:4], a[::-1]       # 10, 50, [20 30 40], dibalik
M = np.arange(1, 13).reshape(3, 4)
M[1, 2]          # baris 1, kolom 2 → 7
M[0]             # baris pertama
M[:, 1]          # kolom ke-1 (semua baris)
M[:2, 1:3]       # sub-matriks
M[[0, 2]]        # fancy index: baris 0 dan 2"""),
            dict(t="Boolean mask & np.where", code=r"""
nilai = np.array([55, 78, 90, 42, 66])
nilai[nilai >= 60]                       # filter
nilai[(nilai >= 60) & (nilai < 80)]      # & | ~ , WAJIB pakai kurung
(nilai >= 60).sum()                      # jumlah yang True
(nilai >= 60).mean()                     # proporsi
np.where(nilai >= 60, 'Lulus', 'Gagal')  # if-else vektor
np.where(nilai < 50)[0]                  # indeks yang memenuhi
np.any(nilai > 85), np.all(nilai > 40)"""),
            dict(t="View vs copy", code=r"""
x = np.arange(5)
v = x[1:3]       # VIEW: mengubah v ikut mengubah x
v[0] = 99        # x → [0 99 2 3 4]
c = x[1:3].copy()  # salinan independen
# fancy index & boolean mask selalu menghasilkan copy""", note="Sumber bug klasik: mengubah slice tanpa <code>.copy()</code>."),
        ]),
        dict(id="n-axis", title="Agregasi & axis", desc="axis=0 → turun sepanjang baris (hasil per kolom); axis=1 → menyamping (hasil per baris).", cards=[
            dict(t="axis=0 vs axis=1", code=r"""
nilai = np.array([[80, 75, 90],     # baris = mahasiswa
                  [60, 85, 70],     # kolom = mata kuliah
                  [95, 65, 88]])
nilai.mean(axis=1)    # rata-rata per MAHASISWA (per baris) → 3 nilai
nilai.mean(axis=0)    # rata-rata per MATA KULIAH (per kolom) → 3 nilai
nilai.mean()          # rata-rata seluruh elemen
nilai.max(axis=0); nilai.argmax(axis=1)   # argmax = indeks nilai maksimum
nilai.sum(axis=0, keepdims=True)          # bentuk (1, 3), enak untuk broadcasting""",
                 note="Ingat: axis yang disebut adalah axis yang <b>hilang/diringkas</b>."),
            dict(t="Fungsi agregasi", code=r"""
x = np.array([4, 8, 15, 16, 23, 42])
x.sum(), x.mean(), np.median(x), x.min(), x.max()
x.std(), x.std(ddof=1)            # populasi vs sampel
x.var(ddof=1)
np.percentile(x, [25, 50, 75])    # kuartil
x.cumsum(); x.cumprod(); np.diff(x)
x.argmin(), x.argmax()            # posisi
np.round(x.mean(), 2)"""),
            dict(t="Nilai unik, urut, hitung", code=r"""
data = np.array([3, 1, 2, 3, 3, 1])
np.unique(data)                               # [1 2 3]
vals, cnt = np.unique(data, return_counts=True)
vals[cnt.argmax()]                            # modus → 3
np.sort(data)[::-1]                           # urut menurun
np.argsort(data)                              # indeks hasil urut
np.bincount(data)                             # frekuensi 0..max"""),
        ]),
        dict(id="n-ops", title="Operasi & broadcasting", desc="Operasi elemen-per-elemen tanpa loop; ukuran berbeda disesuaikan otomatis bila kompatibel.", cards=[
            dict(t="Operasi elemen & fungsi universal", code=r"""
a = np.array([1, 2, 3]); b = np.array([10, 20, 30])
a + b, a * b, b / a, b // a, b % 7, a ** 2
np.sqrt(a), np.exp(a), np.log(b), np.abs(-a), np.sin(a)
np.round(np.pi, 2); np.clip(b, 15, 25)        # batasi ke [15, 25]
np.maximum(a, 2); np.minimum(a, 2)            # elemen-per-elemen
f = np.vectorize(lambda v: v ** 2 if v > 1 else 0)
f(a)                                          # fungsi Python biasa → vektor"""),
            dict(t="Broadcasting", code=r"""
M = np.arange(6).reshape(2, 3)      # (2, 3)
M + 10                               # skalar → semua elemen
M + np.array([1, 2, 3])              # (3,) → dicocokkan per baris
M + np.array([[100], [200]])         # (2, 1) → dicocokkan per kolom
# normalisasi per kolom (z-score):
Z = (M - M.mean(axis=0)) / M.std(axis=0)""",
                 note="Aturan: bandingkan shape dari kanan; dimensi cocok jika <b>sama</b> atau salah satunya <b>1</b>. Jika tidak → <code>ValueError: operands could not be broadcast</code>."),
            dict(t="Aljabar linear", code=r"""
A = np.array([[2, 1], [1, 3]]); v = np.array([1, 2])
A @ v; np.dot(A, v)            # perkalian matriks (BUKAN A * v)
A @ A.T
np.linalg.inv(A); np.linalg.det(A)
np.linalg.solve(A, v)          # selesaikan Ax = v
np.linalg.norm(v)              # panjang vektor"""),
        ]),
        dict(id="n-stat", title="Statistik deskriptif", desc="Template fungsi ringkasan yang sering diminta (mis. waktu tunggu ojek online).", cards=[
            dict(t="summarize(data)", code=r"""
def summarize(data):
    x = np.asarray(data, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    return {
        'n': x.size, 'mean': round(x.mean(), 2), 'median': round(float(np.median(x)), 2),
        'std': round(x.std(ddof=1), 2), 'min': x.min(), 'max': x.max(),
        'q1': q1, 'q3': q3, 'iqr': q3 - q1,
    }

summarize(np.random.default_rng(1).exponential(5, 100))"""),
            dict(t="Outlier IQR & z-score", code=r"""
x = np.random.default_rng(0).normal(50, 10, 200)
q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier = x[(x < batas_bawah) | (x > batas_atas)]
z = (x - x.mean()) / x.std()
outlier_z = x[np.abs(z) > 3]"""),
            dict(t="Skewness, SEM, confidence interval", code=r"""
from scipy import stats
x = np.random.default_rng(2).exponential(5, 100)
stats.skew(x)                      # > 0 → miring kanan (mean > median)
sem = stats.sem(x)                 # std / sqrt(n), ddof=1
lo, hi = stats.t.interval(0.95, df=len(x) - 1, loc=x.mean(), scale=sem)
# manual (n besar): x.mean() ± 1.96 * x.std(ddof=1) / np.sqrt(len(x))""",
                 note="Miring kanan: mean &gt; median. Simetris: mean ≈ median."),
            dict(t="ddof: populasi vs sampel", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Ukuran</th><th>NumPy</th><th>pandas</th><th>Catatan</th></tr></thead><tbody>
<tr><td>Std populasi (÷ n)</td><td><code>x.std()</code></td><td><code>s.std(ddof=0)</code></td><td>default NumPy</td></tr>
<tr><td>Std sampel (÷ n−1)</td><td><code>x.std(ddof=1)</code></td><td><code>s.std()</code></td><td>default pandas</td></tr>
<tr><td>Median</td><td><code>np.median(x)</code></td><td><code>s.median()</code></td><td>tahan outlier</td></tr>
<tr><td>Persentil</td><td><code>np.percentile(x, 90)</code></td><td><code>s.quantile(.9)</code></td><td>persen vs proporsi</td></tr>
</tbody></table></div>"""),
        ]),
        dict(id="n-sim", title="Simulasi stokastik", desc="Stochastic thinking: jalankan percobaan acak berkali-kali, lalu ringkas hasilnya.", cards=[
            dict(t="Monte Carlo: peluang lewat simulasi", code=r"""
rng = np.random.default_rng(42)
n = 100_000
dadu = rng.integers(1, 7, size=(n, 2))           # lempar 2 dadu n kali
p_jumlah_7 = (dadu.sum(axis=1) == 7).mean()      # ≈ 6/36 = 0.1667
# estimasi π
xy = rng.uniform(-1, 1, size=(n, 2))
pi_hat = 4 * ((xy ** 2).sum(axis=1) <= 1).mean()"""),
            dict(t="Simulasi dengan proporsi (mis. jalur sortir paket)", code=r"""
def simulateSorting(n, seed=42):
    rng = np.random.default_rng(seed)
    jalur = rng.choice(['A', 'B', 'C'], size=n, p=[.5, .3, .2])
    durasi = rng.exponential(scale=2.0, size=n)            # menit per paket
    nama, jumlah = np.unique(jalur, return_counts=True)
    return {'jumlah': dict(zip(nama.tolist(), jumlah.tolist())),
            'persen': dict(zip(nama.tolist(), (jumlah / n * 100).round(2).tolist())),
            'total_durasi': round(durasi.sum(), 2), 'rata_durasi': round(durasi.mean(), 2)}

for n in [10, 100, 10_000]:                 # makin besar n → makin dekat teori
    print(n, simulateSorting(n)['persen'])""", note="<b>Hukum bilangan besar</b>: proporsi hasil simulasi mendekati peluang teoretis saat n membesar."),
            dict(t="Central Limit Theorem", code=r"""
rng = np.random.default_rng(0)
populasi = rng.exponential(5, 100_000)            # sangat miring kanan
rata_sampel = [rng.choice(populasi, 30).mean() for _ in range(2000)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].hist(populasi, bins=50); ax[0].set_title('Populasi (miring)')
ax[1].hist(rata_sampel, bins=40); ax[1].set_title('Rata-rata sampel n=30 (≈ normal)')
plt.show()""", note="Rata-rata sampel mendekati normal dengan std ≈ σ/√n walau populasinya tidak normal."),
            dict(t="Histogram + garis mean/median", code=r"""
data = np.random.default_rng(3).exponential(7, 1000)
plt.hist(data, bins=30, edgecolor='white')
plt.axvline(data.mean(), color='red', ls='--', label=f'mean = {data.mean():.1f}')
plt.axvline(np.median(data), color='green', ls='-', label=f'median = {np.median(data):.1f}')
plt.title('Distribusi waktu tunggu'); plt.xlabel('menit'); plt.ylabel('frekuensi')
plt.legend(); plt.show()"""),
        ]),
        dict(id="n-err", title="Error umum", desc="", cards=[
            dict(t="Jebakan NumPy", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Gejala</th><th>Penyebab</th><th>Perbaikan</th></tr></thead><tbody>
<tr><td><code>operands could not be broadcast together</code></td><td>shape tidak kompatibel</td><td>cek <code>.shape</code>; pakai <code>reshape(-1, 1)</code> atau <code>keepdims=True</code></td></tr>
<tr><td>Hasil per baris padahal ingin per kolom</td><td>salah axis</td><td>per kolom = <code>axis=0</code>, per baris = <code>axis=1</code></td></tr>
<tr><td>Std beda dengan pandas/Excel</td><td>ddof</td><td><code>np.std(x, ddof=1)</code> untuk sampel</td></tr>
<tr><td>Array asli ikut berubah</td><td>slice = view</td><td><code>.copy()</code></td></tr>
<tr><td><code>A * B</code> bukan perkalian matriks</td><td><code>*</code> = elemen per elemen</td><td><code>A @ B</code></td></tr>
<tr><td>Hasil acak beda tiap run</td><td>tanpa seed</td><td><code>rng = np.random.default_rng(42)</code></td></tr>
<tr><td><code>The truth value of an array ... is ambiguous</code></td><td><code>and</code>/<code>or</code>/<code>if arr</code></td><td><code>&amp;</code>, <code>|</code>, kurung; <code>.any()</code>/<code>.all()</code></td></tr>
<tr><td>Array berisi string semua</td><td>list campur tipe</td><td>pisahkan kolom / <code>astype(float)</code></td></tr>
</tbody></table></div>"""),
        ]),
    ],
)

PANDAS = dict(
    id="pandas", label="Pandas", count_label="P07",
    placeholder="Cari: merge, groupby, loc, to_datetime, pivot_table…",
    flow=["Baca", "Cek (shape/dtypes/isna)", "Pilih & filter", "Kolom baru", "Gabung", "Ringkas", "Simpan"],
    fixture=r"""
import numpy as np
import pandas as pd
rng = np.random.default_rng(0)
df = pd.DataFrame({
    'id': range(1, 21),
    'tanggal': pd.date_range('2025-01-01', periods=20, freq='3D').strftime('%d/%m/%Y'),
    'kota': rng.choice(['Bandung', 'Jakarta', 'Surabaya'], 20),
    'kategori': rng.choice(['Kopi', 'Teh', 'Makanan'], 20),
    'harga': rng.integers(15, 40, 20) * 1000.0,
    'qty': rng.integers(1, 5, 20),
    'umur': rng.integers(17, 60, 20).astype(float),
})
df.loc[[2, 5], 'umur'] = np.nan
df_koridor = pd.DataFrame({'kota': ['Bandung', 'Jakarta', 'Surabaya', 'Medan'],
                           'wilayah': ['Barat', 'Barat', 'Timur', 'Sumatra']})
df_a = df[['id', 'kota']].copy(); df_b = df[['id', 'harga']].copy()
import os, tempfile
os.chdir(tempfile.mkdtemp())
df.to_csv('data.csv', index=False)
df.to_csv('sensor_utara.csv', index=False); df.to_csv('sensor_pusat.csv', index=False)
""",
    sections=[
        dict(id="p-io", title="Baca, cek, simpan", desc="Langkah a hampir di setiap soal praktikum.", cards=[
            dict(t="read_csv dengan opsi penting", code=r"""
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=',', na_values=['?', '-', -999])  # placeholder → NaN
# sep=';' untuk CSV Excel Indonesia, encoding='latin-1' bila UnicodeDecodeError
df_kecil = pd.read_csv('data.csv', usecols=['id', 'kota', 'harga'])   # hanya kolom tertentu"""),
            dict(t="Cek cepat", code=r"""
df.head(); df.tail(); df.sample(5, random_state=1)
df.shape                    # (baris, kolom)
print(f'{df.shape[0]} baris, {df.shape[1]} kolom')
df.dtypes; df.info()
df.describe(); df.describe(exclude='number')   # kolom teks/kategori
df.isna().sum()             # missing per kolom
df.nunique(); df['kota'].unique(); df['kota'].value_counts()"""),
            dict(t="Gabung banyak file + kolom sumber", code=r"""
frames = []
for st in ['utara', 'pusat']:
    d = pd.read_csv(f'sensor_{st}.csv')
    d['station'] = st                 # tandai asal data
    frames.append(d)
df_all = pd.concat(frames, ignore_index=True)
df_all.groupby('station').size()"""),
            dict(t="Simpan", code=r"""
df.to_csv('hasil_bersih.csv', index=False)   # index=False agar tidak ada kolom 'Unnamed: 0'
df.to_excel('hasil.xlsx', index=False)        # butuh openpyxl""", test=False),
        ]),
        dict(id="p-select", title="Pilih & filter", desc="loc = berdasarkan label, iloc = berdasarkan posisi.", cards=[
            dict(t="Kolom, loc, iloc", code=r"""
df['harga']; df[['kota', 'harga']]         # Series vs DataFrame
df.loc[0, 'harga']                          # label baris 0, kolom harga
df.loc[df['qty'] > 2, ['kota', 'qty']]      # filter + pilih kolom
df.iloc[0]; df.iloc[:5, :3]; df.iloc[-1]    # posisi
df.loc[df['kota'] == 'Bandung', 'diskon'] = 0.1  # isi/ubah nilai dengan kondisi (aman)"""),
            dict(t="Filter kondisi", code=r"""
df[(df['harga'] > 20000) & (df['kota'] != 'Jakarta')]   # &, |, ~ + kurung
df[df['kota'].isin(['Bandung', 'Surabaya'])]
df[df['umur'].between(20, 30)]              # inklusif
df[df['kategori'].str.contains('kop', case=False, na=False)]
df[df['umur'].isna()]; df[df['umur'].notna()]
df.query('harga > 20000 and qty >= 2')
df.nlargest(3, 'harga'); df.nsmallest(3, 'harga')"""),
            dict(t="Urutkan & peringkat", code=r"""
df.sort_values('harga', ascending=False)
df.sort_values(['kota', 'harga'], ascending=[True, False])
df['rank_harga'] = df['harga'].rank(ascending=False, method='dense')
df['harga'].idxmax()                  # label baris dengan harga maksimum
df.loc[df['harga'].idxmax()]
df.reset_index(drop=True)"""),
        ]),
        dict(id="p-col", title="Kolom baru & transformasi", desc="Feature engineering: aritmetika, kondisi, kategori, teks.", cards=[
            dict(t="Aritmetika & kondisi", code=r"""
df['total'] = df['harga'] * df['qty']
df['mahal'] = df['harga'] > df['harga'].median()          # boolean
df['label'] = np.where(df['qty'] >= 3, 'Banyak', 'Sedikit')
kondisi = [df['harga'] < 20000, df['harga'] < 30000]
df['level'] = np.select(kondisi, ['Murah', 'Sedang'], default='Mahal')
df = df.assign(total_ribu=lambda d: d['total'] / 1000)"""),
            dict(t="apply, map, replace", code=r"""
def kategori_umur(u):
    if pd.isna(u): return 'Tidak diketahui'
    return 'Muda' if u < 30 else 'Dewasa'

df['grup_umur'] = df['umur'].apply(kategori_umur)
df['kota_kode'] = df['kota'].map({'Bandung': 'BDG', 'Jakarta': 'JKT', 'Surabaya': 'SBY'})
df['kategori'] = df['kategori'].replace({'Teh': 'Non-Kopi'})
df['rasio'] = df.apply(lambda r: r['harga'] / r['qty'], axis=1)   # per baris (lambat)"""),
            dict(t="Binning: cut & qcut", code=r"""
df['umur_bin'] = pd.cut(df['umur'], bins=[0, 25, 40, 100],
                        labels=['Muda', 'Dewasa', 'Senior'], right=False)
df['harga_level'] = pd.qcut(df['harga'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  # kuartil
# kategori ISPU dari pm25 → pd.cut dengan batas resmi""", note="<code>cut</code> = batas ditentukan; <code>qcut</code> = jumlah anggota per bin sama."),
            dict(t="Teks (.str)", code=r"""
s = pd.Series(['  jakarta', 'BANDUNG ', 'Action|Drama', '120 min'])
s.str.strip().str.title()                  # rapikan
s.str.split('|').str[0]                    # genre pertama
s.str.extract(r'(\d+)', expand=False).astype(float)   # ambil angka
s.str.replace('min', '', regex=False)
s.str.len(); s.str.startswith('B'); s.str.lower()"""),
            dict(t="Tipe, rename, drop", code=r"""
df['kota'] = df['kota'].astype('category')
df['harga'] = pd.to_numeric(df['harga'], errors='coerce')     # teks rusak → NaN
df.rename(columns={'qty': 'jumlah'})          # hasilnya DataFrame baru → assign agar permanen
df = df.drop(columns=['rank_harga'])
df.columns = df.columns.str.lower().str.replace(' ', '_')
kolom_teks = df.select_dtypes(include=['object', 'string', 'category']).columns"""),
        ]),
        dict(id="p-time", title="Datetime", desc="Ubah ke datetime dulu, baru ambil komponen dengan .dt.", cards=[
            dict(t="to_datetime & komponen", code=r"""
df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d/%m/%Y')   # DD/MM/YYYY
df['tahun'] = df['tanggal'].dt.year
df['bulan'] = df['tanggal'].dt.month          # 1-12
df['nama_hari'] = df['tanggal'].dt.day_name()
df['hari_ke'] = df['tanggal'].dt.day_of_week  # Senin=0 ... Minggu=6
df['is_weekend'] = df['hari_ke'] >= 5
df['jenis_hari'] = np.where(df['is_weekend'], 'Akhir Pekan', 'Hari Kerja')
# timestamp lengkap: .dt.hour, .dt.date, .dt.to_period('M')""",
                 note="Tanpa <code>format</code>, '01/02/2025' bisa terbaca 2 Januari. Di pandas 3, <code>dt.weekday</code>/<code>dayofweek</code> diganti <code>dt.day_of_week</code>."),
            dict(t="Filter tanggal, resample, hari lengkap", code=r"""
df[(df['tanggal'] >= '2025-02-01') & (df['tanggal'] < '2025-03-01')]
harian = df.groupby('tanggal')['harga'].sum()
mingguan = harian.resample('W').sum()           # butuh DatetimeIndex
bulanan = df.set_index('tanggal')['harga'].resample('MS').mean()
semua_hari = pd.date_range(harian.index.min(), harian.index.max(), freq='D')
harian = harian.reindex(semua_hari, fill_value=0)   # hari kosong = 0
harian.rolling(7).mean()                         # rata-rata bergerak 7 hari"""),
        ]),
        dict(id="p-group", title="Groupby, pivot & crosstab", desc="Split → apply → combine.", cards=[
            dict(t="groupby + agg bernama", code=r"""
df.groupby('kota')['harga'].mean()
df.groupby('kota').agg(total=('harga', 'sum'),
                       rata=('harga', 'mean'),
                       n_transaksi=('id', 'count'),
                       hari_unik=('tanggal', 'nunique')).sort_values('total', ascending=False)
df.groupby(['kota', 'kategori'])['harga'].sum().unstack(fill_value=0)   # jadi tabel lebar
df.groupby('kota').size()        # jumlah baris (termasuk NaN)"""),
            dict(t="transform: nilai grup ke tiap baris", code=r"""
df['rata_kota'] = df.groupby('kota')['harga'].transform('mean')
df['pct_dari_kota'] = df['harga'] / df.groupby('kota')['harga'].transform('sum')
df['umur'] = df['umur'].fillna(df.groupby('kota')['umur'].transform('median'))  # imputasi per grup"""),
            dict(t="pivot_table, crosstab, value_counts", code=r"""
pd.pivot_table(df, index='bulan', columns='kota', values='harga', aggfunc='sum', fill_value=0)
pd.crosstab(df['kota'], df['kategori'])                       # frekuensi
pd.crosstab(df['kota'], df['kategori'], normalize='index')     # proporsi per baris (=1)
df['kota'].value_counts(normalize=True).mul(100).round(1)      # persentase
df.melt(id_vars='id', value_vars=['harga', 'qty'])            # lebar → panjang"""),
        ]),
        dict(id="p-merge", title="Merge, join & concat", desc="Pilih jenis join sesuai baris mana yang wajib tetap ada.", cards=[
            dict(t="Jenis merge", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>how</th><th>Baris yang dipertahankan</th><th>Kapan dipakai</th></tr></thead><tbody>
<tr><td><code>inner</code> (default)</td><td>key ada di kedua tabel</td><td>hanya data lengkap (mis. film yang punya data box office)</td></tr>
<tr><td><code>left</code></td><td>semua baris tabel kiri</td><td>"semua koridor tetap muncul" → master di kiri, NaN untuk yang tidak cocok</td></tr>
<tr><td><code>right</code></td><td>semua baris tabel kanan</td><td>kebalikan left</td></tr>
<tr><td><code>outer</code></td><td>semua baris kedua tabel</td><td>mencari data yang tidak berpasangan (pakai <code>indicator=True</code>)</td></tr>
</tbody></table></div>"""),
            dict(t="merge", code=r"""
df_gab = df.merge(df_koridor, on='kota', how='left')
# nama key berbeda:
a = df_a.rename(columns={'id': 'ID'})
pd.merge(a, df_b, left_on='ID', right_on='id', how='inner').drop(columns='ID')
# cek hasil:
cek = df_koridor.merge(df[['kota']].drop_duplicates(), on='kota', how='outer', indicator=True)
cek['_merge'].value_counts()     # left_only = tidak punya pasangan""",
                 note="Jumlah baris bertambah setelah merge? Key tidak unik (many-to-many). Cek dengan <code>validate='one_to_one'</code>/<code>'many_to_one'</code>."),
            dict(t="concat", code=r"""
pd.concat([df_a, df_a], ignore_index=True)          # tumpuk baris
pd.concat([df_a, df_b['harga']], axis=1)            # tempel kolom (berdasarkan index)
df_a.join(df_b.set_index('id'), on='id')            # join via index"""),
        ]),
        dict(id="p-miss", title="Missing value & duplikat (ringkas)", desc="Detail lengkap ada di tab Cleaning Data.", cards=[
            dict(t="Pola cepat", code=r"""
df.isna().sum(); df.isna().mean().mul(100).round(1)    # jumlah & persen
df = df.dropna(subset=['kota'])                         # buang jika kolom penting kosong
df['umur'] = df['umur'].fillna(df['umur'].median())
num = df.select_dtypes('number').columns
df[num] = df[num].fillna(df[num].median())              # semua kolom numerik sekaligus
df['kota'] = df['kota'].fillna(df['kota'].mode()[0])
df = df.replace(-999, np.nan)
df.duplicated().sum(); df = df.drop_duplicates().reset_index(drop=True)"""),
        ]),
        dict(id="p-err", title="Error & warning umum", desc="", cards=[
            dict(t="Jebakan pandas", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Gejala</th><th>Penyebab</th><th>Perbaikan</th></tr></thead><tbody>
<tr><td><code>SettingWithCopyWarning</code> / nilai tidak berubah</td><td>chained assignment <code>df[df.a&gt;0]['b'] = 1</code></td><td><code>df.loc[df.a &gt; 0, 'b'] = 1</code>; subset baru pakai <code>.copy()</code></td></tr>
<tr><td><code>KeyError: 'kolom'</code></td><td>salah nama / spasi tersembunyi</td><td><code>df.columns.tolist()</code>, <code>df.columns.str.strip()</code></td></tr>
<tr><td><code>Can only use .dt accessor with datetimelike values</code></td><td>kolom masih string</td><td><code>pd.to_datetime(..., format=...)</code> dulu</td></tr>
<tr><td><code>could not convert string to float</code></td><td>ada "Rp", titik ribuan, "?"</td><td><code>str.replace</code> lalu <code>pd.to_numeric(errors='coerce')</code></td></tr>
<tr><td>Baris membengkak setelah merge</td><td>key duplikat</td><td><code>drop_duplicates</code> key / <code>validate=</code></td></tr>
<tr><td>Tidak ada perubahan setelah <code>fillna</code>/<code>drop</code></td><td>lupa assign</td><td><code>df = df.dropna()</code> (hindari <code>inplace</code>)</td></tr>
<tr><td><code>ValueError: The truth value of a Series is ambiguous</code></td><td><code>and</code>/<code>or</code></td><td><code>&amp;</code>/<code>|</code> + kurung</td></tr>
<tr><td>Kolom teks bertipe <code>str</code> (pandas 3)</td><td>dtype string default baru</td><td>pilih dengan <code>select_dtypes(include=['object', 'string'])</code></td></tr>
</tbody></table></div>"""),
        ]),
    ],
)

SKLEARN = dict(
    id="sklearn", label="Scikit-learn", count_label="P09",
    placeholder="Cari: train_test_split, Pipeline, f1, GridSearchCV, KMeans…",
    flow=["X, y", "train_test_split", "Pipeline (prep + model)", "fit", "predict", "Evaluasi", "CV / tuning"],
    fixture=r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame({'luas': rng.uniform(36, 300, n).round(), 'umur': rng.integers(0, 40, n).astype(float),
                   'kota': rng.choice(['Jakarta', 'Bandung', 'Surabaya'], n),
                   'sertifikat': rng.choice(['SHM', 'HGB'], n)})
df['harga'] = 300 + 8 * df['luas'] - 10 * df['umur'] + np.where(df['kota'] == 'Jakarta', 600, 0) + rng.normal(0, 150, n)
df['churn'] = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.02 * df['umur'] - 0.004 * df['luas'])))).astype(int)
df.loc[rng.choice(n, 15, replace=False), 'umur'] = np.nan
df.loc[rng.choice(n, 10, replace=False), 'kota'] = np.nan
""",
    sections=[
        dict(id="s-alur", title="Konsep & API", desc="Semua estimator scikit-learn memakai pola yang sama.", cards=[
            dict(t="Pola fit / predict / transform", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Method</th><th>Dipakai pada</th><th>Fungsi</th><th>Data</th></tr></thead><tbody>
<tr><td><code>fit(X, y)</code></td><td>model &amp; transformer</td><td>belajar parameter (koefisien, mean/std, kategori)</td><td><b>train saja</b></td></tr>
<tr><td><code>predict(X)</code></td><td>model</td><td>prediksi label / nilai</td><td>train &amp; test</td></tr>
<tr><td><code>predict_proba(X)</code></td><td>classifier</td><td>peluang tiap kelas (<code>[:, 1]</code> = kelas positif)</td><td>test</td></tr>
<tr><td><code>transform(X)</code></td><td>scaler, encoder, imputer, PCA</td><td>terapkan parameter hasil fit</td><td>train &amp; test</td></tr>
<tr><td><code>fit_transform(X)</code></td><td>transformer</td><td>fit + transform sekaligus</td><td><b>train saja</b></td></tr>
<tr><td><code>score(X, y)</code></td><td>model</td><td>R² (regresi) / accuracy (klasifikasi)</td><td>test</td></tr>
</tbody></table></div>"""),
            dict(t="Jenis masalah → model", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Masalah</th><th>Target</th><th>Model umum</th><th>Metrik</th></tr></thead><tbody>
<tr><td><span class="chip">Regresi</span></td><td>angka kontinu (harga)</td><td><code>LinearRegression</code>, <code>Ridge</code>, <code>DecisionTreeRegressor</code>, <code>RandomForestRegressor</code>, <code>KNeighborsRegressor</code></td><td>MAE, RMSE, R²</td></tr>
<tr><td><span class="chip">Klasifikasi</span></td><td>kategori (churn 0/1)</td><td><code>LogisticRegression</code>, <code>KNeighborsClassifier</code>, <code>DecisionTreeClassifier</code>, <code>RandomForestClassifier</code>, <code>GaussianNB</code>, <code>SVC</code></td><td>accuracy, precision, recall, F1, AUC</td></tr>
<tr><td><span class="chip">Clustering</span></td><td>tanpa label</td><td><code>KMeans</code>, <code>AgglomerativeClustering</code>, <code>DBSCAN</code></td><td>inertia, silhouette</td></tr>
<tr><td><span class="chip">Reduksi dimensi</span></td><td>tanpa label</td><td><code>PCA</code></td><td>explained variance</td></tr>
</tbody></table></div>"""),
            dict(t="Import yang sering dipakai", code=r"""
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics"""),
        ]),
        dict(id="s-split", title="X, y & train-test split", desc="Split dulu sebelum preprocessing apa pun.", cards=[
            dict(t="Siapkan X dan y", code=r"""
X = df.drop(columns=['harga', 'churn'])     # fitur (DataFrame 2D)
y = df['harga']                             # target (1D)
# dari array 1D (satu fitur) → wajib 2D:
x = np.arange(10)
X1 = x.reshape(-1, 1)                       # (10, 1)
X1 = df[['luas']]                           # kurung ganda → DataFrame 2D""", note="<code>Expected 2D array, got 1D array</code> → pakai <code>reshape(-1, 1)</code> atau <code>df[['kolom']]</code>."),
            dict(t="train_test_split", code=r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
# klasifikasi: jaga proporsi kelas
yc = df['churn']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, yc, test_size=0.25, stratify=yc, random_state=42)
yc_train.mean(), yc_test.mean()             # proporsi hampir sama"""),
        ]),
        dict(id="s-prep", title="Preprocessing, ColumnTransformer & Pipeline", desc="Pipeline mencegah data leakage: fit hanya di train, juga di setiap fold CV.", cards=[
            dict(t="Transformer satu per satu", code=r"""
scaler = StandardScaler()                         # mean 0, std 1
Xs_train = scaler.fit_transform(X_train[['luas']])  # fit di TRAIN
Xs_test = scaler.transform(X_test[['luas']])        # test cukup transform
MinMaxScaler().fit_transform(X_train[['luas']])     # skala 0-1
SimpleImputer(strategy='median').fit_transform(X_train[['umur']])  # mean/median/most_frequent/constant
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit_transform(X_train[['sertifikat']]); ohe.get_feature_names_out()
OrdinalEncoder(categories=[['HGB', 'SHM']]).fit_transform(X_train[['sertifikat']])"""),
            dict(t="ColumnTransformer + Pipeline (template utama)", wide=True, code=r"""
fitur_num = X.select_dtypes(include='number').columns.tolist()
fitur_cat = X.select_dtypes(exclude='number').columns.tolist()

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', num_pipe, fitur_num),
                                  ('cat', cat_pipe, fitur_cat)])   # kolom lain dibuang (remainder='drop')

pipe = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
pipe.named_steps['prep'].get_feature_names_out()   # nama fitur setelah one-hot
pipe.named_steps['model'].coef_""",
                 note="Ganti model cukup di langkah <code>'model'</code>. Hyperparameter di dalam pipeline diakses dengan <code>model__nama_param</code>."),
            dict(t="make_pipeline & output pandas", code=r"""
p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))   # nama step otomatis
p.fit(X_train[['luas']], yc.loc[X_train.index])
pre = ColumnTransformer([('num', StandardScaler(), ['luas'])]).set_output(transform='pandas')
pre.fit_transform(X_train).head()          # hasil transform tetap DataFrame"""),
        ]),
        dict(id="s-reg", title="Regresi", desc="Prediksi angka kontinu.", cards=[
            dict(t="LinearRegression & koefisien", code=r"""
lr = LinearRegression().fit(X_train[['luas']], y_train)
lr.coef_, lr.intercept_           # y = coef*x + intercept
print(f'y = {lr.coef_[0]:.2f} x + {lr.intercept_:.2f}')
lr.predict(pd.DataFrame({'luas': [100, 200]}))"""),
            dict(t="Metrik regresi", code=r"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)                       # sklearn ≥1.4: root_mean_squared_error
r2 = r2_score(y_test, y_pred)
print(f'MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}')""",
                 note="MAE &amp; RMSE bersatuan target (makin kecil makin baik). R² = 1 sempurna, 0 = setara tebak rata-rata, &lt; 0 = lebih buruk dari rata-rata."),
            dict(t="Polinomial & overfitting", code=r"""
xp = np.sort(rng.uniform(-3, 3, 40)).reshape(-1, 1)
yp = 0.5 * xp[:, 0]**3 - xp[:, 0]**2 + rng.normal(0, 1.5, 40)
a, b, c, d = train_test_split(xp, yp, test_size=0.3, random_state=42)
for deg in [1, 3, 15]:
    m = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression()).fit(a, c)
    print(deg, round(m.score(a, c), 3), round(m.score(b, d), 3))   # R² train vs test"""),
            dict(t="Plot aktual vs prediksi & residual", code=r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(y_test, y_pred, alpha=.6)
lim = [y_test.min(), y_test.max()]
ax[0].plot(lim, lim, 'r--', label='y = x'); ax[0].legend()
ax[0].set(title='Aktual vs Prediksi', xlabel='Aktual', ylabel='Prediksi')
sns.histplot(y_test - y_pred, kde=True, ax=ax[1])
ax[1].set(title='Residual', xlabel='Aktual − Prediksi')
plt.tight_layout(); plt.show()"""),
            dict(t="Model pohon & feature importance", code=r"""
rf = Pipeline([('prep', preprocessor),
               ('model', RandomForestRegressor(n_estimators=200, random_state=42))])
rf.fit(X_train, y_train)
imp = pd.Series(rf.named_steps['model'].feature_importances_,
                index=rf.named_steps['prep'].get_feature_names_out()).sort_values()
imp.plot(kind='barh', title='Feature importance'); plt.show()"""),
        ]),
        dict(id="s-clf", title="Klasifikasi", desc="Prediksi kelas; jangan hanya melihat accuracy bila kelas tidak seimbang.", cards=[
            dict(t="Latih & prediksi", code=r"""
clf = Pipeline([('prep', preprocessor), ('model', LogisticRegression(max_iter=1000))])
clf.fit(Xc_train, yc_train)
pred = clf.predict(Xc_test)
proba = clf.predict_proba(Xc_test)[:, 1]          # peluang kelas 1
pred_03 = (proba >= 0.3).astype(int)              # threshold lebih rendah → recall naik
# model lain: KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier(max_depth=4, random_state=42),
#             RandomForestClassifier(n_estimators=200, random_state=42)"""),
            dict(t="Metrik klasifikasi", code=r"""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
accuracy_score(yc_test, pred)
precision_score(yc_test, pred, zero_division=0); recall_score(yc_test, pred); f1_score(yc_test, pred)
print(classification_report(yc_test, pred, zero_division=0))
confusion_matrix(yc_test, pred)             # [[TN, FP], [FN, TP]]
ConfusionMatrixDisplay.from_predictions(yc_test, pred, cmap='Blues'); plt.grid(False); plt.show()
roc_auc_score(yc_test, proba)              # pakai PROBA, bukan label"""),
            dict(t="Rumus & kapan dipakai", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Metrik</th><th>Rumus</th><th>Pertanyaan</th><th>Penting bila</th></tr></thead><tbody>
<tr><td>Accuracy</td><td>(TP+TN) / semua</td><td>berapa persen tebakan benar?</td><td>kelas seimbang</td></tr>
<tr><td>Precision</td><td>TP / (TP+FP)</td><td>dari yang ditebak positif, berapa yang benar?</td><td>false positive mahal (spam, vonis)</td></tr>
<tr><td>Recall</td><td>TP / (TP+FN)</td><td>dari positif sebenarnya, berapa yang tertangkap?</td><td>false negative mahal (penyakit, churn, fraud)</td></tr>
<tr><td>F1</td><td>2·P·R / (P+R)</td><td>keseimbangan P dan R</td><td>kelas tidak seimbang</td></tr>
<tr><td>ROC-AUC</td><td>luas di bawah kurva TPR–FPR</td><td>seberapa baik memisahkan kelas di semua threshold</td><td>membandingkan model</td></tr>
</tbody></table></div>
<p class="note">Baseline: tebak kelas mayoritas terus → accuracy = proporsi mayoritas, recall = 0. Model harus mengalahkan baseline.</p>"""),
            dict(t="ROC curve", code=r"""
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(yc_test, proba, name='LogReg')
plt.plot([0, 1], [0, 1], 'k--', label='acak'); plt.legend()
plt.title('ROC Curve'); plt.show()"""),
        ]),
        dict(id="s-cv", title="Cross-validation & tuning", desc="Evaluasi yang lebih stabil dan memilih hyperparameter tanpa menyentuh test set.", cards=[
            dict(t="cross_val_score", code=r"""
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skor = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
print(skor.round(3), skor.mean().round(3), skor.std().round(3))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(clf, X, yc, cv=skf, scoring='f1')
# scoring lain: 'accuracy', 'recall', 'roc_auc', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'"""),
            dict(t="GridSearchCV", code=r"""
pipe_tree = Pipeline([('prep', preprocessor), ('model', DecisionTreeClassifier(random_state=42))])
param_grid = {'model__max_depth': [2, 3, 4, 6, 8],
              'model__min_samples_leaf': [1, 5, 10]}
grid = GridSearchCV(pipe_tree, param_grid, cv=skf, scoring='f1')
grid.fit(Xc_train, yc_train)
grid.best_params_, grid.best_score_
best = grid.best_estimator_                       # sudah di-refit di seluruh train
f1_score(yc_test, best.predict(Xc_test), zero_division=0)
pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score']].sort_values('mean_test_score')"""),
            dict(t="Kurva validasi manual (mis. nilai k KNN)", code=r"""
ks = range(1, 30, 2); hasil = []
for k in ks:
    m = Pipeline([('prep', preprocessor), ('model', KNeighborsClassifier(n_neighbors=k))])
    hasil.append(cross_val_score(m, Xc_train, yc_train, cv=skf, scoring='f1').mean())
plt.plot(list(ks), hasil, marker='o'); plt.xlabel('k'); plt.ylabel('F1 (CV)')
plt.title('F1 vs k'); plt.show()""", note="k kecil → overfitting (variance tinggi); k besar → underfitting (bias tinggi). KNN wajib scaling karena berbasis jarak."),
        ]),
        dict(id="s-cluster", title="Clustering & PCA", desc="Unsupervised: tidak ada y.", cards=[
            dict(t="KMeans + elbow + silhouette", code=r"""
from sklearn.metrics import silhouette_score
fitur = ['luas', 'harga']
Xk = StandardScaler().fit_transform(df[fitur])      # scaling wajib
inertia, sil = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xk)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xk, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(range(2, 9), inertia, marker='o'); ax[0].set(title='Elbow', xlabel='k', ylabel='inertia')
ax[1].plot(range(2, 9), sil, marker='o'); ax[1].set(title='Silhouette', xlabel='k')
plt.show()""", note="Elbow: titik tekuk penurunan inertia. Silhouette: −1..1, makin tinggi makin terpisah."),
            dict(t="Model final & profil cluster", code=r"""
scaler = StandardScaler(); Xk = scaler.fit_transform(df[fitur])
km = KMeans(n_clusters=3, n_init=10, random_state=42)
df['cluster'] = km.fit_predict(Xk)
df.groupby('cluster')[fitur].mean().round(1).assign(n=df['cluster'].value_counts())
centroid = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=fitur)  # skala asli
sns.scatterplot(data=df, x='luas', y='harga', hue='cluster', palette='tab10')
plt.scatter(centroid['luas'], centroid['harga'], marker='X', s=200, c='black'); plt.show()"""),
            dict(t="PCA untuk visualisasi 2D", code=r"""
pca = PCA(n_components=2, random_state=42)
komp = pca.fit_transform(Xk)
pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum()
sns.scatterplot(x=komp[:, 0], y=komp[:, 1], hue=df['cluster'], palette='tab10')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()"""),
        ]),
        dict(id="s-teori", title="Overfitting, leakage & teori", desc="Bahan soal analisis dan teori.", cards=[
            dict(t="Underfitting vs overfitting", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Kondisi</th><th>Skor train</th><th>Skor test</th><th>Penyebab</th><th>Solusi</th></tr></thead><tbody>
<tr><td>Underfitting (bias tinggi)</td><td>rendah</td><td>rendah</td><td>model terlalu sederhana</td><td>model lebih kompleks, tambah fitur, kurangi regularisasi</td></tr>
<tr><td>Good fit</td><td>tinggi</td><td>tinggi, dekat train</td><td>kompleksitas pas</td><td>—</td></tr>
<tr><td>Overfitting (variance tinggi)</td><td>sangat tinggi</td><td>jauh lebih rendah</td><td>menghafal noise</td><td>sederhanakan (max_depth, derajat), regularisasi, data lebih banyak, CV</td></tr>
</tbody></table></div>"""),
            dict(t="Data leakage", html=r"""
<ul>
<li>Scaling/imputasi/encoding di-<code>fit</code> pada seluruh data sebelum split.</li>
<li>Memakai fitur yang baru diketahui setelah kejadian (mis. <code>tanggal_berhenti</code> untuk prediksi churn).</li>
<li>Target ikut masuk ke <code>X</code> (lupa <code>drop</code>).</li>
<li>Tuning hyperparameter dengan melihat skor test → pakai CV di data train.</li>
</ul><p class="note"><b>Solusi:</b> split dulu → semua preprocessing dalam <code>Pipeline</code> → test set hanya dipakai sekali di akhir.</p>"""),
            dict(t="Jawaban singkat yang sering ditanya", html=r"""
<ul>
<li><b>Kenapa scaling?</b> Model berbasis jarak/gradien (KNN, KMeans, LogReg, SVM) dikuasai fitur berskala besar. Model pohon tidak perlu scaling.</li>
<li><b>Kenapa one-hot, bukan label encoding?</b> Fitur nominal tidak punya urutan; angka 0/1/2 membuat urutan palsu.</li>
<li><b>Kenapa stratify?</b> Proporsi kelas train/test sama dengan data asli.</li>
<li><b>Kenapa random_state?</b> Hasil bisa direproduksi.</li>
<li><b>MAE vs RMSE?</b> RMSE lebih menghukum kesalahan besar.</li>
<li><b>Supervised vs unsupervised?</b> Ada label (regresi/klasifikasi) vs tanpa label (clustering/PCA).</li>
</ul>"""),
        ]),
        dict(id="s-err", title="Error umum", desc="", cards=[
            dict(t="Pesan error scikit-learn", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Pesan</th><th>Penyebab</th><th>Perbaikan</th></tr></thead><tbody>
<tr><td><code>Expected 2D array, got 1D array</code></td><td>X berupa Series/array 1D</td><td><code>x.reshape(-1, 1)</code> / <code>df[['kolom']]</code></td></tr>
<tr><td><code>could not convert string to float: 'Jakarta'</code></td><td>kolom kategorikal belum di-encode</td><td><code>OneHotEncoder</code> dalam <code>ColumnTransformer</code></td></tr>
<tr><td><code>Input X contains NaN</code></td><td>missing value</td><td><code>SimpleImputer</code> di pipeline</td></tr>
<tr><td><code>Found unknown categories ... during transform</code></td><td>kategori baru di test</td><td><code>OneHotEncoder(handle_unknown='ignore')</code></td></tr>
<tr><td><code>ConvergenceWarning: lbfgs failed to converge</code></td><td>iterasi kurang / tidak di-scale</td><td><code>LogisticRegression(max_iter=1000)</code> + <code>StandardScaler</code></td></tr>
<tr><td><code>Found input variables with inconsistent numbers of samples</code></td><td>panjang X dan y beda</td><td>cek <code>X.shape</code>, <code>y.shape</code>; urutan output <code>train_test_split</code></td></tr>
<tr><td><code>The least populated class in y has only 1 member</code></td><td>stratify dengan kelas sangat sedikit / y kontinu</td><td>stratify hanya untuk klasifikasi</td></tr>
<tr><td><code>Invalid parameter 'max_depth' for estimator Pipeline</code></td><td>nama param grid salah</td><td><code>'model__max_depth'</code> (nama step + 2 underscore)</td></tr>
<tr><td><code>X has feature names, but ... was fitted without feature names</code> (warning)</td><td>fit dengan array, predict dengan DataFrame</td><td>konsisten pakai DataFrame</td></tr>
</tbody></table></div>"""),
        ]),
    ],
)

# Extra section for existing Visualisasi tab (P08 practicum recipes)
P08_SECTION = dict(
    id="p08", title="Resep Praktikum 08", desc="Pola yang muncul di soal P08: compound plot, subplot vertikal, stacked bar dari pivot, heatmap dengan sel kosong, dan checklist fungsi seaborn.",
    fixture=r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rng = np.random.default_rng(0)
tgl = pd.date_range('2025-01-01', '2025-12-31', freq='D')
df_bus = pd.DataFrame({'tanggal': np.repeat(tgl, 3), 'id_koridor': np.tile(['K01', 'K02', 'K03'], len(tgl))})
df_bus['jam'] = rng.integers(5, 23, len(df_bus))
for c in ['pnp_umum', 'pnp_pelajar', 'pnp_lansia']:
    df_bus[c] = rng.integers(0, 100, len(df_bus))
df_bus = df_bus[~((df_bus['id_koridor'] == 'K03') & (df_bus['tanggal'] < '2025-07-01'))]
df_bus['tanggal'] = df_bus['tanggal'].dt.strftime('%d/%m/%Y')
df_koridor = pd.DataFrame({'id_koridor': ['K01', 'K02', 'K03', 'K04'], 'wilayah': ['Utara', 'Timur', 'Selatan', 'Barat']})
df_volley = pd.DataFrame({'match_id': np.repeat([f'M{i:03d}' for i in range(30)], 2),
                          'team': rng.choice(list('ABCDEF'), 60), 'result': ['win', 'lose'] * 30,
                          'kills': rng.integers(20, 70, 60), 'attack_errors': rng.integers(3, 25, 60),
                          'unforced_errors': rng.integers(5, 35, 60)})
df_cars = pd.DataFrame({
    'km_per_liter': rng.normal(10, 3, 200), 'weight_kg': rng.normal(1300, 300, 200),
    'origin': rng.choice(['usa', 'europe', 'japan'], 200), 'cylinders': rng.choice([4, 6, 8], 200),
    'horsepower': rng.normal(100, 30, 200), 'model_year': rng.integers(70, 83, 200)})
""",
    cards=[
        dict(t="Compound plot fungsi (np.vectorize)", code=r"""
x = np.arange(0, 6, 0.25)                         # [0, 6) langkah 0.25
f_sin = np.vectorize(lambda v: np.sin(v))
f_exp = np.vectorize(lambda v: np.exp(v / 2))     # e^x terlalu besar → kurva lain tampak datar
f_sqrt = np.vectorize(lambda v: np.sqrt(v))
fig, ax = plt.subplots(figsize=(8, 5))
for f, nama in [(f_sin, 'Sinus'), (f_exp, 'Eksponensial'), (f_sqrt, 'Akar Kuadrat')]:
    ax.plot(x, f(x), marker='.', label=nama)
ax.set(title='Compound Plot', xlabel='x', ylabel='f(x)'); ax.legend(); ax.grid(alpha=.5)
plt.show()"""),
        dict(t="Dua subplot vertikal (sharex)", code=r"""
a1 = [1.5, 2.9, 4.2, 6.8, 9.1, 13.4, 18.0, 24.7, 33.5, 45.2]
a2 = [10.0, 8.7, 6.1, 4.2, 3.5, 3.9, 5.6, 7.4, 9.8, 11.0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(a1, color='tab:blue', label='First Array'); ax1.set(title='Timeseries #1', ylabel='f(t)'); ax1.legend()
ax2.plot(a2, color='tab:orange', label='Second Array'); ax2.set(title='Timeseries #2', xlabel='t', ylabel='f(t)'); ax2.legend()
plt.tight_layout(); plt.show()"""),
        dict(t="Bukti pola: fit eksponensial & kuadratik", code=r"""
t = np.arange(len(a1)); a1, a2 = np.array(a1), np.array(a2)
print(np.round(a1[1:] / a1[:-1], 2))          # rasio ≈ konstan → eksponensial
b, ln_a = np.polyfit(t, np.log(a1), 1)        # ln y = ln a + b t
coef = np.polyfit(t, a2, 2)                   # parabola → kuadratik
r2 = lambda y, yh: 1 - ((y - yh)**2).sum() / ((y - y.mean())**2).sum()
r2(a1, np.exp(ln_a + b * t)), r2(a2, np.polyval(coef, t))""",
             note="Naik makin curam + rasio konstan → <b>eksponensial</b>. Turun lalu naik (bentuk U) → <b>kuadratik</b>."),
        dict(t="Kolom total, bulan, jenis hari → line chart per tanggal", code=r"""
df_bus['tanggal'] = pd.to_datetime(df_bus['tanggal'], format='%d/%m/%Y')
df_bus['total_pnp'] = df_bus[['pnp_umum', 'pnp_pelajar', 'pnp_lansia']].sum(axis=1)
df_bus['bulan'] = df_bus['tanggal'].dt.month
df_bus['jenis_hari'] = np.where(df_bus['tanggal'].dt.day_of_week >= 5, 'Akhir Pekan', 'Hari Kerja')
harian = df_bus.groupby('tanggal')['total_pnp'].sum()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(harian.index, harian.values, lw=1)
ax.set(title='Total penumpang per tanggal', xlabel='Tanggal', ylabel='Penumpang'); plt.show()""",
             note="Garis bergerigi teratur tiap 7 hari = pola mingguan (hari kerja vs akhir pekan). Lonjakan mendadak → cek koridor baru di tabel ringkasan."),
        dict(t="Ringkasan per id + left merge (semua tetap muncul)", code=r"""
df_ringkas = df_bus.groupby('id_koridor').agg(total_pnp=('total_pnp', 'sum'),
                                              hari_operasi=('tanggal', 'nunique')).reset_index()
df_kor_ringkas = df_koridor.merge(df_ringkas, on='id_koridor', how='left')   # master di KIRI
df_kor_ringkas[['total_pnp', 'hari_operasi']] = df_kor_ringkas[['total_pnp', 'hari_operasi']].fillna(0).astype(int)
df_kor_ringkas"""),
        dict(t="Stacked bar dari pivot_table", code=r"""
gab = df_bus.merge(df_koridor[['id_koridor', 'wilayah']], on='id_koridor', how='left')
pv = gab.pivot_table(index='bulan', columns='wilayah', values='total_pnp', aggfunc='sum').fillna(0)
ax = pv.plot(kind='bar', stacked=True, figsize=(10, 5), width=.8)
ax.set_xticklabels(['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'], rotation=0)
ax.set(title='Penumpang per bulan per wilayah', xlabel='Bulan', ylabel='Total')
ax.legend(title='Wilayah', bbox_to_anchor=(1.01, 1), loc='upper left'); plt.tight_layout(); plt.show()"""),
        dict(t="Dua garis dari groupby + unstack", code=r"""
rata = df_bus.groupby(['jam', 'jenis_hari'])['total_pnp'].mean().unstack('jenis_hari')
ax = rata.plot(marker='o', figsize=(9, 4))
ax.set_xticks(rata.index)
ax.set(title='Rata-rata penumpang per jam', xlabel='Jam', ylabel='Rata-rata total_pnp')
ax.legend(title='Jenis hari'); plt.show()
rata.idxmax()                                   # jam puncak tiap jenis hari"""),
        dict(t="Heatmap pivot dengan sel kosong", code=r"""
hm = (df_bus.pivot_table(index='id_koridor', columns='bulan', values='pnp_pelajar', aggfunc='mean')
      .reindex(df_koridor['id_koridor']))         # koridor tanpa data tetap tampil (baris kosong)
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(hm, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5, ax=ax)
ax.grid(False)                                    # sel NaN tidak tertimpa garis grid tema
ax.set(title='Rata-rata pelajar per koridor per bulan', xlabel='Bulan', ylabel='Koridor'); plt.show()""",
             note="Sel kosong = NaN = tidak ada data (koridor belum beroperasi). Penurunan Juni–Juli pada pelajar = libur sekolah."),
        dict(t="Top-10 & bottom-10 (barh berdampingan)", code=r"""
df_volley['KER'] = df_volley['kills'] / df_volley['attack_errors']
df_volley['label'] = df_volley['match_id'] + ' – ' + df_volley['team']
top = df_volley.nlargest(10, 'KER').sort_values('KER')
bot = df_volley.nsmallest(10, 'KER').sort_values('KER', ascending=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
a.barh(top['label'], top['KER'], color='tab:green'); a.set(title='10 KER tertinggi', xlabel='KER')
b.barh(bot['label'], bot['KER'], color='tab:red'); b.set(title='10 KER terendah', xlabel='KER')
plt.tight_layout(); plt.show()"""),
        dict(t="Satu match dua baris → satu baris (filter + merge)", code=r"""
df_win = df_volley.loc[df_volley['result'] == 'win', ['match_id', 'unforced_errors']] \
                  .rename(columns={'unforced_errors': 'errors_winner'})
df_lose = df_volley.loc[df_volley['result'] == 'lose', ['match_id', 'unforced_errors']] \
                   .rename(columns={'unforced_errors': 'errors_loser'})
df_game = df_win.merge(df_lose, on='match_id')
(df_game['errors_winner'] < df_game['errors_loser']).mean()     # proporsi pemenang lebih sedikit error
sns.scatterplot(data=df_game, x='errors_loser', y='errors_winner')
lim = [0, 40]; plt.plot(lim, lim, 'k--', label='y = x'); plt.legend(); plt.show()"""),
        dict(t="Checklist fungsi seaborn P08", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Tugas</th><th>Kode</th><th>Yang dibaca</th></tr></thead><tbody>
<tr><td>Distribusi + KDE</td><td><code>sns.histplot(data=df, x='km_per_liter', kde=True)</code></td><td>bentuk, skew, puncak</td></tr>
<tr><td>Jumlah per kategori + hue</td><td><code>sns.countplot(data=df, x='cylinders', hue='origin')</code></td><td>komposisi tiap kelompok</td></tr>
<tr><td>Hubungan 2 numerik</td><td><code>sns.scatterplot(data=df, x='weight_kg', y='km_per_liter', hue='origin')</code></td><td>arah, kekuatan, klaster</td></tr>
<tr><td>Distribusi per kelompok</td><td><code>sns.violinplot(data=df, x='origin', y='km_per_liter', inner='quartile')</code></td><td>median, sebaran, bentuk</td></tr>
<tr><td>Rata-rata + CI 95%</td><td><code>sns.barplot(data=df, x='origin', y='km_per_liter', errorbar=('ci', 95))</code></td><td>CI tidak beririsan → beda meyakinkan</td></tr>
<tr><td>Kuartil &amp; outlier</td><td><code>sns.boxplot(data=df, x='cylinders', y='horsepower')</code></td><td>median, IQR, titik outlier</td></tr>
<tr><td>Matriks korelasi</td><td><code>sns.heatmap(df.select_dtypes('number').corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)</code></td><td>korelasi terkuat dengan target</td></tr>
<tr><td>Density satu grup</td><td><code>sns.kdeplot(data=df[df.origin == 'japan'], x='km_per_liter', fill=True)</code></td><td>puncak &amp; rentang</td></tr>
<tr><td>Garis regresi</td><td><code>sns.regplot(data=df, x='model_year', y='km_per_liter', x_jitter=.25)</code> / <code>sns.lmplot(..., hue='origin')</code></td><td>tren naik/turun</td></tr>
</tbody></table></div>
<p class="note">seaborn 0.13: <code>palette</code> tanpa <code>hue</code> memicu warning → tambahkan <code>hue=x, legend=False</code>. <code>ci=95</code> sudah diganti <code>errorbar=('ci', 95)</code>.</p>"""),
        dict(t="Uji cepat checklist seaborn", code=r"""
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
sns.histplot(data=df_cars, x='km_per_liter', kde=True, ax=ax[0, 0])
sns.countplot(data=df_cars, x='cylinders', hue='origin', ax=ax[0, 1])
sns.scatterplot(data=df_cars, x='weight_kg', y='km_per_liter', hue='origin', ax=ax[0, 2])
sns.violinplot(data=df_cars, x='origin', y='km_per_liter', hue='origin', legend=False, ax=ax[1, 0])
sns.barplot(data=df_cars, x='origin', y='km_per_liter', errorbar=('ci', 95), capsize=.1, ax=ax[1, 1])
sns.regplot(data=df_cars, x='model_year', y='km_per_liter', scatter_kws={'alpha': .4}, ax=ax[1, 2])
plt.tight_layout(); plt.show()"""),
    ],
)


def run_tests():
    import os, warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bad = 0
    for name, cards, fixture in [
        ("numpy", [c for s in NUMPY["sections"] for c in s["cards"]], NUMPY["fixture"]),
        ("pandas", [c for s in PANDAS["sections"] for c in s["cards"]], PANDAS["fixture"]),
        ("sklearn", [c for s in SKLEARN["sections"] for c in s["cards"]], SKLEARN["fixture"]),
        ("p08", P08_SECTION["cards"], P08_SECTION["fixture"]),
    ]:
        ns = {}
        cwd = os.getcwd()
        exec(fixture, ns)
        for c in cards:
            if "code" not in c or c.get("test") is False:
                continue
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    exec(c["code"].strip("\n"), ns)
                except Exception as e:
                    bad += 1
                    print(f"[{name}] FAIL {c['t']}: {type(e).__name__}: {e}")
                for x in w:
                    msg = str(x.message)
                    if "FigureCanvasAgg is non-interactive" in msg:
                        continue
                    print(f"[{name}] WARN {c['t']}: {x.category.__name__}: {msg[:160]}")
            plt.close("all")
        os.chdir(cwd)
        print(f"[{name}] tested")
    return bad

if __name__ == "__main__":
    import sys
    sys.exit(1 if run_tests() else 0)
