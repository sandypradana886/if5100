"""Build the 'Prediksi Soal' tab: run every answer for real, capture printed output and charts,
then write tools/prediksi.json (used by build_page.py) and two notebooks in public/notebook/.

Usage (from the project root):
    pip install -r tools/requirements.txt
    python tools/make_prediksi.py
"""
import contextlib, io, json, re, textwrap, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import markdown as md_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
NB_DIR = ROOT / "public" / "notebook"
warnings.filterwarnings("ignore")


def d(s):
    return textwrap.dedent(s).strip("\n")


# =====================================================================================
# CONTENT
# =====================================================================================
RULES = d("""
- Kerjakan di **Google Colab** (runtime CPU). Jalankan sel **berurutan** dari atas ke bawah, karena tiap soal memakai hasil soal sebelumnya.
- Library yang boleh dipakai: `numpy`, `pandas`, `matplotlib`, `seaborn`.
- **Jangan ubah** nama fungsi, parameter, dan kode uji. Output grafik tidak harus 100% sama, tetapi nilai yang dikembalikan fungsi harus benar.
- Setiap grafik wajib punya **judul, label sumbu, dan legend** (jika ada lebih dari satu seri).
""")

SCENARIO = d("""
Kamu adalah analis data di jaringan kedai **Kopi Senja** yang punya 4 cabang: Bandung, Jakarta, Surabaya, dan Yogyakarta.
Manajer memberikan data transaksi **September–Oktober 2026**. Data ini diambil dari 4 sistem kasir yang berbeda,
sehingga **kotor**: penulisan nama cabang tidak seragam, harga tersimpan sebagai teks (`"Rp 27.000"`, `"?"`),
ada umur pelanggan yang mustahil, rating kosong, transaksi ganda, dan kolom yang isinya sama semua.

Tugasmu: **membersihkan data**, lalu menjawab pertanyaan bisnis manajer dengan **visualisasi** yang tepat.
""")

SCENARIO_EN = ("You are a data analyst at the Kopi Senja coffee chain (4 branches). The September–October 2026 "
               "transaction data is messy. Clean it, then answer the manager's business questions with the right visualizations.")

COLUMNS = [
    ("id_transaksi", "ID unik transaksi", "identifier"),
    ("tanggal", "tanggal transaksi (teks `YYYY-MM-DD`)", "temporal"),
    ("cabang", "nama cabang (penulisan tidak seragam)", "kategorikal nominal"),
    ("kategori", "Kopi / Non-Kopi / Makanan", "kategorikal nominal"),
    ("harga", "harga satuan, **teks**: `27000`, `Rp 27.000`, atau `?`", "seharusnya numerik"),
    ("qty", "jumlah item", "numerik diskret"),
    ("metode_bayar", "QRIS / Tunai / Kartu", "kategorikal nominal"),
    ("umur_pelanggan", "umur pelanggan (ada NaN dan nilai mustahil)", "numerik"),
    ("durasi_tunggu", "lama menunggu pesanan (menit)", "numerik kontinu"),
    ("rating", "rating 1–5 (ada NaN)", "ordinal"),
    ("pajak_persen", "persentase pajak", "konstan"),
]

SETUP = d('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 120)


def buat_dataset(seed=8, n=640):
    """Data transaksi Kopi Senja (Sep-Okt 2026) yang sengaja dibuat kotor. JANGAN DIUBAH."""
    rng = np.random.default_rng(seed)
    tanggal = pd.Timestamp('2026-09-01') + pd.to_timedelta(rng.integers(0, 61, n), unit='D')
    cabang = rng.choice(['Bandung', 'Jakarta', 'Surabaya', 'Yogyakarta'], n, p=[.30, .34, .21, .15])
    kategori = rng.choice(['Kopi', 'Non-Kopi', 'Makanan'], n, p=[.50, .22, .28])
    dasar = pd.Series({'Kopi': 27000, 'Non-Kopi': 23000, 'Makanan': 33000})
    harga = dasar[kategori].to_numpy() + rng.integers(-4, 5, n) * 1000
    akhir_pekan = tanggal.dayofweek >= 5
    qty = rng.integers(1, 4, n) + akhir_pekan * rng.integers(0, 3, n)
    metode = rng.choice(['QRIS', 'Tunai', 'Kartu'], n, p=[.55, .25, .20])
    umur = np.clip(rng.normal(27, 7, n), 16, 65).round()
    durasi = (rng.gamma(2.2, 3.2, n) + np.where(cabang == 'Jakarta', 4, 0)).round(1)
    rating = np.clip(np.round(5.2 - 0.13 * durasi + rng.normal(0, .6, n)), 1, 5)

    df = pd.DataFrame({
        'id_transaksi': [f'TRX{i:04d}' for i in range(1, n + 1)],
        'tanggal': tanggal.strftime('%Y-%m-%d'), 'cabang': cabang, 'kategori': kategori,
        'harga': harga.astype(str), 'qty': qty, 'metode_bayar': metode,
        'umur_pelanggan': umur, 'durasi_tunggu': durasi, 'rating': rating, 'pajak_persen': 11,
    })
    # ---- membuat data kotor ----
    i = rng.choice(n, 70, replace=False)
    df.loc[i[:30], 'cabang'] = df.loc[i[:30], 'cabang'].str.lower()
    df.loc[i[30:50], 'cabang'] = df.loc[i[30:50], 'cabang'].str.upper()
    df.loc[i[50:], 'cabang'] = ' ' + df.loc[i[50:], 'cabang'] + ' '
    j = rng.choice(n, 160, replace=False)
    df.loc[j[:140], 'harga'] = ['Rp ' + f'{int(h):,}'.replace(',', '.') for h in df.loc[j[:140], 'harga']]
    df.loc[j[140:], 'harga'] = '?'
    k = rng.choice(n, 40, replace=False)
    df.loc[k[:6], 'umur_pelanggan'] = [999, 0, 150, 999, 2, 120]
    df.loc[k[6:], 'umur_pelanggan'] = np.nan
    df.loc[rng.choice(n, 45, replace=False), 'rating'] = np.nan
    dup = df.sample(18, random_state=seed)
    return pd.concat([df, dup]).sample(frac=1, random_state=seed).reset_index(drop=True)


df_raw = buat_dataset()
print(df_raw.shape)
print(df_raw.head(8).to_string())
''')

SOALS = []


def soal(no, judul, poin, topik, teks, teks_en, kerangka, jawaban, penjelasan):
    SOALS.append(dict(no=no, judul=judul, poin=poin, topik=topik, teks=d(teks), teks_en=teks_en,
                      kerangka=d(kerangka), jawaban=d(jawaban), penjelasan=d(penjelasan)))


soal(1, "Pembersihan data", 20, "cleaning", """
Implementasikan `bersihkanData(df_raw)` yang mengembalikan **DataFrame baru** yang sudah bersih (jangan ubah `df_raw`).
Lakukan langkah berikut **berurutan**:

1. **Seragamkan** `cabang`: hapus spasi di awal/akhir dan ubah ke *Title Case* (`" jakarta"` → `"Jakarta"`).
2. Ubah `harga` menjadi **numerik**: buang `"Rp"`, spasi, dan titik ribuan; nilai `"?"` menjadi `NaN`.
3. **Hapus baris duplikat**, lalu reset index.
4. Isi `harga` yang kosong dengan **median harga per kategori**.
5. `umur_pelanggan` di luar rentang **15–80** dianggap tidak valid → jadikan `NaN`, lalu isi semua `NaN` dengan **median**.
6. Isi `rating` yang kosong dengan **median**.
7. Ubah `tanggal` menjadi `datetime`.
8. **Buang kolom konstan** (hanya punya 1 nilai unik).
9. Tambahkan kolom `total = harga * qty` dan `akhir_pekan` (`True` jika Sabtu/Minggu).
""", "Implement bersihkanData(df_raw): standardize branch names, convert price text to numbers, drop duplicates, impute price by category median, invalidate ages outside 15–80 and impute with the median, impute rating, parse dates, drop constant columns, add total and akhir_pekan.",
"""
def bersihkanData(df_raw):
    df = df_raw.copy()
    # TODO: tulis langkah 1-9 di sini
    return df


# ---- KODE UJI (jangan diubah) ----
df = bersihkanData(df_raw)
print('Ukuran awal :', df_raw.shape)
print('Ukuran akhir:', df.shape)
print('Cabang      :', sorted(df['cabang'].unique()))
print('Missing     :', int(df.isna().sum().sum()))
print('Rentang umur:', df['umur_pelanggan'].min(), '-', df['umur_pelanggan'].max())
print(df.dtypes)
""", """
def bersihkanData(df_raw):
    df = df_raw.copy()
    # 1) seragamkan teks cabang
    df['cabang'] = df['cabang'].str.strip().str.title()
    # 2) harga teks -> angka ('Rp 27.000' -> 27000, '?' -> NaN)
    df['harga'] = pd.to_numeric(df['harga'].str.replace(r'[^0-9]', '', regex=True), errors='coerce')
    # 3) duplikat (setelah teks seragam, sebelum imputasi)
    df = df.drop_duplicates().reset_index(drop=True)
    # 4) harga kosong -> median per kategori
    df['harga'] = df['harga'].fillna(df.groupby('kategori')['harga'].transform('median'))
    # 5) umur tidak valid -> NaN -> median
    df.loc[~df['umur_pelanggan'].between(15, 80), 'umur_pelanggan'] = np.nan
    df['umur_pelanggan'] = df['umur_pelanggan'].fillna(df['umur_pelanggan'].median())
    # 6) rating kosong -> median
    df['rating'] = df['rating'].fillna(df['rating'].median())
    # 7) tanggal -> datetime
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    # 8) buang kolom konstan
    df = df.drop(columns=df.columns[df.nunique() <= 1])
    # 9) fitur baru
    df['total'] = df['harga'] * df['qty']
    df['akhir_pekan'] = df['tanggal'].dt.dayofweek >= 5
    return df


# ---- KODE UJI (jangan diubah) ----
df = bersihkanData(df_raw)
print('Ukuran awal :', df_raw.shape)
print('Ukuran akhir:', df.shape)
print('Cabang      :', sorted(df['cabang'].unique()))
print('Missing     :', int(df.isna().sum().sum()))
print('Rentang umur:', df['umur_pelanggan'].min(), '-', df['umur_pelanggan'].max())
print(df.dtypes)
""", """
- **Urutan penting.** Teks cabang diseragamkan dulu agar `" Jakarta "` dan `"jakarta"` dianggap sama, lalu duplikat dibuang **sebelum** imputasi supaya median tidak dihitung dari baris ganda.
- `str.replace(r'[^0-9]', '')` membuang semua karakter selain angka. `"?"` menjadi string kosong, lalu `to_numeric(errors='coerce')` mengubahnya ke `NaN`.
- Harga diisi **median per kategori** karena harga Makanan, Kopi, dan Non-Kopi berbeda jauh. Median keseluruhan akan memberi harga yang salah untuk tiap kategori.
- Umur 0, 2, 120, 150, dan 999 mustahil secara domain, jadi diperlakukan sebagai *missing*, bukan dibuang barisnya (transaksinya tetap valid).
- `pajak_persen` bernilai 11 di semua baris (variansi 0) sehingga tidak informatif dan dibuang.
- Hasil: **{n_raw} → {n_bersih} baris** ({n_dup} duplikat dibuang), 0 missing value, 4 nama cabang.
""")

soal(2, "Penjualan per cabang", 15, "categorical", """
Manajer bertanya: **cabang mana yang penjualannya paling besar?**

1. Implementasikan `penjualanPerCabang(df)` yang mengembalikan DataFrame ber-index `cabang` dengan kolom
   `total_penjualan` (jumlah `total`), `jumlah_transaksi`, dan `rata_per_transaksi`, **terurut menurun** berdasarkan `total_penjualan`.
2. Buat **bar chart horizontal** total penjualan (dalam juta rupiah) dengan label nilai di ujung batang.
""", "Implement penjualanPerCabang(df) returning total sales, number of transactions and average per transaction by branch (sorted descending), then draw a labelled horizontal bar chart.",
"""
def penjualanPerCabang(df):
    # TODO
    pass


ringkas = penjualanPerCabang(df)
print(ringkas)

# TODO: bar chart horizontal total penjualan per cabang (juta Rp) + label nilai
""", """
def penjualanPerCabang(df):
    hasil = (df.groupby('cabang')
               .agg(total_penjualan=('total', 'sum'), jumlah_transaksi=('id_transaksi', 'count'))
               .sort_values('total_penjualan', ascending=False))
    hasil['rata_per_transaksi'] = (hasil['total_penjualan'] / hasil['jumlah_transaksi']).round()
    return hasil


ringkas = penjualanPerCabang(df)
print(ringkas)

urut = ringkas.sort_values('total_penjualan')          # terbesar di atas
fig, ax = plt.subplots(figsize=(8, 4))
bar = ax.barh(urut.index, urut['total_penjualan'] / 1e6, color='#1f77b4')
ax.bar_label(bar, fmt='%.1f jt', padding=3)
ax.set_title('Total Penjualan per Cabang (Sep-Okt 2026)', weight='bold')
ax.set_xlabel('total penjualan (juta Rp)')
ax.set_xlim(0, urut['total_penjualan'].max() / 1e6 * 1.15)
plt.tight_layout()
plt.show()
""", """
- **Bar chart** dipilih karena membandingkan nilai antar **kategori** memakai *position/length*, channel paling akurat. Diurutkan agar ranking langsung terbaca, dan horizontal agar nama cabang mudah dibaca.
- Sumbu x dimulai dari **0**. Panjang batang adalah nilainya, jadi memotong sumbu akan melebih-lebihkan selisih.
- **{top_cabang}** tertinggi dengan Rp {top_total:.1f} juta dari {top_trx} transaksi; **{low_cabang}** terendah (Rp {low_total:.1f} juta).
""")

soal(3, "Distribusi durasi tunggu", 15, "categorical", """
Pelanggan mengeluh soal antrean. Analisis **durasi tunggu**:

1. Implementasikan `statistikDurasi(df)` yang mengembalikan dictionary `{'mean', 'median', 'skew', 'n_outlier'}`
   (mean & median dibulatkan 2 desimal, skew 3 desimal, outlier memakai **aturan IQR 1,5×**).
2. Buat 2 panel berdampingan: **histogram** `durasi_tunggu` dengan garis vertikal mean & median, dan **boxplot** `durasi_tunggu` per cabang.
""", "Implement statistikDurasi(df) (mean, median, skewness, IQR outlier count) and plot a histogram with mean/median lines next to a boxplot by branch.",
"""
def statistikDurasi(df):
    # TODO
    pass


stat = statistikDurasi(df)
print(stat)

# TODO: subplot 1x2 -> histogram + garis mean/median | boxplot per cabang
""", """
def statistikDurasi(df):
    s = df['durasi_tunggu']
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    return {'mean': round(float(s.mean()), 2), 'median': round(float(s.median()), 2),
            'skew': round(float(s.skew()), 3), 'n_outlier': int(n_out)}


stat = statistikDurasi(df)
print(stat)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(df['durasi_tunggu'], bins=25, color='#1f77b4', edgecolor='white')
axs[0].axvline(stat['mean'], color='red', ls='--', label=f"mean {stat['mean']}")
axs[0].axvline(stat['median'], color='green', ls='--', label=f"median {stat['median']}")
axs[0].set(title='Distribusi Durasi Tunggu', xlabel='menit', ylabel='jumlah transaksi')
axs[0].legend()
sns.boxplot(data=df, x='cabang', y='durasi_tunggu', color='#9ecae1', ax=axs[1])
axs[1].set(title='Durasi Tunggu per Cabang', xlabel='', ylabel='menit')
plt.tight_layout()
plt.show()
""", """
- Skewness **{skew}** (positif) → distribusi **miring ke kanan**: kebanyakan pesanan selesai cepat, tetapi ada ekor panjang pesanan yang sangat lama. Karena itu **mean ({mean}) > median ({median})**.
- Ada **{n_out}** transaksi outlier menurut aturan IQR. Ini kemungkinan **nyata** (antrean panjang), bukan galat input, jadi tidak dibuang.
- Boxplot memperlihatkan **Jakarta** paling lama (median {med_jkt} menit vs {med_lain} menit di cabang lain): seluruh kotaknya bergeser ke atas, jadi yang lambat bukan hanya beberapa pesanan, tetapi layanannya secara umum.
- Histogram menjawab "bagaimana bentuk sebarannya", sedangkan boxplot memudahkan **membandingkan sebaran antar grup** sekaligus menandai outlier.
""")

soal(4, "Komposisi kategori produk", 15, "hierarchical", """
Apakah komposisi produk yang terjual **berbeda antar cabang**?

1. Implementasikan `komposisiKategori(df)` yang mengembalikan tabel **proporsi** transaksi tiap `kategori` per `cabang`
   (baris = cabang, kolom = kategori, **setiap baris berjumlah 1**, dibulatkan 3 desimal).
2. Visualisasikan dengan **100% stacked bar chart** horizontal, lengkap dengan label persentase di tiap segmen.
""", "Implement komposisiKategori(df) returning the share of each product category per branch (rows sum to 1), then draw a labelled 100% stacked bar chart.",
"""
def komposisiKategori(df):
    # TODO
    pass


komp = komposisiKategori(df)
print(komp)

# TODO: 100% stacked bar chart + label persentase
""", """
def komposisiKategori(df):
    return pd.crosstab(df['cabang'], df['kategori'], normalize='index').round(3)


komp = komposisiKategori(df)
print(komp)

ax = komp.plot(kind='barh', stacked=True, figsize=(9, 4), width=.7,
               color=['#8c564b', '#ff7f0e', '#2ca02c'])
for c in ax.containers:
    ax.bar_label(c, labels=[f'{v:.0%}' for v in c.datavalues], label_type='center',
                 color='white', fontsize=8)
ax.set(title='Komposisi Kategori Produk per Cabang', xlabel='proporsi transaksi', ylabel='')
ax.legend(title='kategori', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
""", """
- `pd.crosstab(..., normalize='index')` langsung menghitung proporsi per baris (bagian terhadap keseluruhan tiap cabang).
- **100% stacked bar** dipakai karena yang ditanya adalah **komposisi (part-to-whole) yang dibandingkan antar cabang**. Empat pie chart berdampingan sulit dibandingkan karena mata harus menilai sudut, channel yang kurang akurat.
- Kopi mendominasi di semua cabang ({kopi_min:.0%}–{kopi_max:.0%}). **{kopi_top}** paling tinggi porsi kopinya ({kopi_max:.0%}); tiga cabang lainnya hampir identik (sekitar {kopi_min:.0%}–{kopi_rest_max:.0%}).
""")

soal(5, "Tren penjualan harian", 15, "temporal", """
Manajer ingin melihat **tren penjualan harian** dan pengaruh **akhir pekan**.

1. Implementasikan `penjualanHarian(df)` yang mengembalikan **Series** total penjualan per tanggal. Index harus berupa
   tanggal **lengkap setiap hari** (hari tanpa transaksi bernilai 0).
2. Buat **line chart** penjualan harian (juta Rp) beserta **rata-rata bergerak 7 hari**, dan tandai titik akhir pekan.
3. Cetak rata-rata penjualan per hari untuk hari kerja vs akhir pekan.
""", "Implement penjualanHarian(df) returning a daily sales Series with every calendar day, plot it with a 7-day moving average and highlight weekends, then compare weekday vs weekend averages.",
"""
def penjualanHarian(df):
    # TODO
    pass


harian = penjualanHarian(df)
print(harian.describe().round(0))

# TODO: line chart harian + rolling 7 hari + tandai akhir pekan
# TODO: rata-rata penjualan per hari: hari kerja vs akhir pekan
""", """
def penjualanHarian(df):
    harian = df.groupby('tanggal')['total'].sum()
    return harian.asfreq('D', fill_value=0)      # tanggal tanpa transaksi -> 0


harian = penjualanHarian(df)
print(harian.describe().round(0))
ma7 = harian.rolling(7).mean()
weekend = harian[harian.index.dayofweek >= 5]

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(harian.index, harian / 1e6, color='#1f77b4', alpha=.45, marker='o', ms=3, label='harian')
ax.plot(ma7.index, ma7 / 1e6, color='#ff7f0e', lw=2.2, label='rata-rata bergerak 7 hari')
ax.scatter(weekend.index, weekend / 1e6, color='#d62728', s=18, zorder=3, label='akhir pekan')
ax.set(title='Penjualan Harian Kopi Senja', ylabel='juta Rp')
ax.legend()
plt.tight_layout()
plt.show()

per_hari = harian.groupby(harian.index.dayofweek >= 5).mean() / 1e6
print(per_hari.rename({False: 'hari kerja', True: 'akhir pekan'}).round(2))
""", """
- `asfreq('D', fill_value=0)` memastikan sumbu waktu kontinu. Tanpa itu, hari tanpa transaksi hilang dan garis langsung "melompat".
- Titik merah (akhir pekan) umumnya berada di **puncak** ({we_above:.0%} hari akhir pekan di atas median hari kerja). Rata-rata penjualan akhir pekan **Rp {we:.2f} juta/hari** vs hari kerja **Rp {wd:.2f} juta/hari** (sekitar **{ratio:.1f}×**).
- **Rata-rata bergerak 7 hari** meratakan pola mingguan sehingga tren jangka panjang terlihat: bergerak di kisaran **Rp {ma_min:.2f}–{ma_max:.2f} juta/hari** tanpa tren naik atau turun yang jelas.
- Line chart tidak wajib mulai dari 0, karena yang dibaca adalah pola naik-turun.
""")

soal(6, "Durasi tunggu vs rating", 10, "relational", """
Apakah **menunggu lama membuat rating turun**?

1. Implementasikan `korelasiDurasiRating(df)` yang mengembalikan koefisien korelasi Pearson antara `durasi_tunggu` dan `rating` (3 desimal).
2. Buat 2 panel: **scatter plot** durasi vs rating dengan garis regresi (beri *jitter* karena rating berupa bilangan bulat), dan **heatmap korelasi** kolom numerik.
""", "Implement korelasiDurasiRating(df) (Pearson r, 3 decimals), then plot a jittered scatter with a regression line and a correlation heatmap of the numeric columns.",
"""
def korelasiDurasiRating(df):
    # TODO
    pass


r = korelasiDurasiRating(df)
print('r(durasi_tunggu, rating) =', r)

# TODO: scatter + garis regresi | heatmap korelasi
""", """
def korelasiDurasiRating(df):
    return round(float(df['durasi_tunggu'].corr(df['rating'])), 3)


r = korelasiDurasiRating(df)
print('r(durasi_tunggu, rating) =', r)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
sns.regplot(data=df, x='durasi_tunggu', y='rating', y_jitter=.15, ax=axs[0],
            scatter_kws={'alpha': .35, 's': 15}, line_kws={'color': 'red'})
axs[0].set(title=f'Durasi Tunggu vs Rating (r = {r})', xlabel='durasi tunggu (menit)')
num = df[['harga', 'qty', 'total', 'umur_pelanggan', 'durasi_tunggu', 'rating']]
sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axs[1])
axs[1].set_title('Matriks Korelasi')
plt.tight_layout()
plt.show()
""", """
- r = **{r}** → korelasi **negatif {kekuatan}**: makin lama menunggu, rating cenderung makin rendah.
- Garis regresi menurun jelas. *Jitter* dipakai karena rating hanya bernilai 1–5, sehingga tanpa jitter titik-titiknya bertumpuk.
- Di heatmap, `total` berkorelasi sangat kuat dengan `qty` ({r_tq}) tetapi hanya {r_th} dengan `harga`: `total = harga × qty`, dan variasi qty (1–5 item) jauh lebih besar daripada variasi harga. `umur_pelanggan` hampir tidak berkorelasi dengan apa pun.
- **Korelasi bukan sebab-akibat.** Tetapi bersama temuan Soal 3 (Jakarta menunggu paling lama), ini memberi dasar untuk menambah barista di Jakarta, lalu dievaluasi ulang.
""")

soal(7, "Perbaiki kode visualisasi", 10, "debug", """
Kode di bawah ditulis mengikuti contoh lama dan mengandung **4 kesalahan** (error, peringatan *deprecated*, atau hasil yang menyesatkan).
Temukan dan perbaiki semuanya, lalu jelaskan tiap perbaikan.
""", "The code below contains 4 mistakes (errors, deprecated calls or misleading output). Find and fix them, then explain each fix.",
"""
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.distplot(df['umur_pelanggan'], kde=False, bins=10, ax=axs[0])
axs[0].set_title('Umur Pelanggan')

axs[1].hist(df['durasi_tunggu'], bins=20, normed=True)
axs[1].set_title('Durasi Tunggu')

bayar = df['metode_bayar'].value_counts()
warna = plt.get_cmap('Set2')(np.linspace(0, 1, 2))
axs[2].pie(bayar, labels=bayar.index, autopct='%1.0f%%', colors=warna)
axs[2].set_title('Metode Bayar')
plt.show()

print(df.corr())
""", """
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df['umur_pelanggan'], bins=10, ax=axs[0])                 # (1) distplot -> histplot
axs[0].set_title('Umur Pelanggan')

axs[1].hist(df['durasi_tunggu'], bins=20, density=True)                # (2) normed -> density
axs[1].set(title='Durasi Tunggu', ylabel='density')

bayar = df['metode_bayar'].value_counts()
warna = plt.get_cmap('Set2')(np.linspace(0, 1, len(bayar)))           # (3) jumlah warna = jumlah kategori
axs[2].pie(bayar, labels=bayar.index, autopct='%1.0f%%', colors=warna)
axs[2].set_title('Metode Bayar')
plt.tight_layout()
plt.show()

print(df.corr(numeric_only=True).round(2))                             # (4) numeric_only
""", """
1. `sns.distplot` sudah **deprecated** sejak seaborn 0.11 → ganti `sns.histplot` (tanpa KDE adalah default).
2. `normed=True` sudah **dihapus** dari matplotlib (`AttributeError`) → ganti `density=True`.
3. Hanya 2 warna untuk 3 kategori, sehingga **warna berulang** dan dua irisan tampak sama (menyesatkan) → jumlah warna = `len(bayar)`.
4. `df.corr()` **error** di pandas ≥ 2 karena ada kolom teks (`cabang`, `kategori`, …) → `df.corr(numeric_only=True)`.
""")

ANALISIS = [
    ("Berdasarkan Soal 2, apakah cabang dengan total penjualan tertinggi juga yang nilai rata-rata per transaksinya tertinggi? Apa artinya?",
     "Based on Problem 2, does the branch with the highest total sales also have the highest average value per transaction? What does that imply?",
     "Tidak harus. **{top_cabang}** unggul karena **jumlah transaksinya paling banyak** ({top_trx}). Rata-rata per transaksi semua cabang hampir sama (Rp {avg_min:,.0f}–{avg_max:,.0f}), "
     "sehingga perbedaan total penjualan lebih ditentukan oleh **volume pelanggan**, bukan besarnya belanja per orang. Untuk menaikkan penjualan cabang kecil, fokusnya menarik lebih banyak pelanggan."),
    ("Mengapa nilai `durasi_tunggu` dan `umur_pelanggan` yang kosong sebaiknya diisi dengan median, bukan mean?",
     "Why should missing values be filled with the median rather than the mean?",
     "Karena distribusinya **tidak simetris atau mengandung outlier**. `durasi_tunggu` miring ke kanan (skew {skew}), dan `umur_pelanggan` awalnya berisi nilai mustahil seperti 999. "
     "Mean tertarik ke arah nilai ekstrem, sedangkan **median robust** (tidak terpengaruh outlier), sehingga lebih mewakili nilai tipikal."),
    ("Manajer meminta 4 pie chart (satu per cabang) untuk komposisi kategori. Apakah itu pilihan terbaik? Jelaskan dengan konsep visual encoding.",
     "The manager asks for 4 pie charts (one per branch). Is that the best choice? Explain using visual-encoding concepts.",
     "Kurang tepat. Pie chart mengandalkan **angle/area**, yang menurut ranking akurasi persepsi berada di bawah **position/length**. "
     "Membandingkan irisan antar 4 pie yang terpisah juga sulit (yang dibandingkan harus berdekatan dan memakai skala sama). "
     "**100% stacked bar** menaruh semua cabang pada satu sumbu yang sama sehingga perbedaan proporsi langsung terlihat."),
    ("Apa fungsi rata-rata bergerak 7 hari pada Soal 5, dan mengapa jendelanya 7?",
     "What is the purpose of the 7-day moving average in Problem 5, and why a window of 7?",
     "Penjualan punya **pola mingguan** (puncak tiap akhir pekan). Jendela **7 hari** selalu memuat tepat satu siklus minggu (5 hari kerja + 2 hari akhir pekan), "
     "sehingga efek akhir pekan teredam dan **tren jangka panjang** terlihat. Jendela lain (misalnya 5) akan memuat jumlah akhir pekan yang berubah-ubah, sehingga garisnya tetap bergelombang."),
    ("Berdasarkan Soal 3 dan Soal 6, rekomendasi apa yang kamu berikan ke manajer? Sertakan bukti dari grafik.",
     "Based on Problems 3 and 6, what would you recommend to the manager? Cite evidence from the charts.",
     "Prioritaskan **mempercepat layanan di cabang Jakarta**. Buktinya: median durasi tunggu Jakarta {med_jkt} menit vs {med_lain} menit di cabang lain (boxplot Soal 3), "
     "dan durasi tunggu berkorelasi negatif dengan rating (r = {r}, Soal 6). Karena korelasi bukan sebab-akibat, rekomendasi ini perlu diuji, "
     "misalnya menambah barista pada jam sibuk, lalu bandingkan rating sebelum dan sesudahnya."),
]

TEORI = [
    ("Apa beda visualisasi *exploratory* dan *explanatory*? Beri contoh dari studi kasus Kopi Senja.",
     "**Exploratory** dipakai analis untuk menemukan pola: mencoba banyak grafik, misalnya histogram semua kolom atau heatmap korelasi. "
     "**Explanatory** dipakai untuk menyampaikan satu pesan ke audiens, misalnya satu bar chart \"Jakarta paling lama menunggu\" untuk rapat manajer."),
    ("Urutkan channel berikut dari yang paling akurat untuk data kuantitatif: *area, color, position, angle, length*.",
     "**position → length → angle → area → color** (Spoerri; Mackinlay). Karena itu bar chart (position/length) lebih akurat daripada pie (angle) atau bubble (area)."),
    ("Mengapa color hue (merah, hijau, biru) kurang tepat untuk data ordinal seperti rating 1–5? Apa alternatifnya?",
     "Hue **tidak punya urutan alami** di persepsi manusia, sehingga pembaca tidak tahu apakah hijau lebih tinggi dari biru. "
     "Untuk ordinal, pakai **saturation/lightness** (gradasi terang–gelap satu warna, misalnya `cmap='Blues'`) atau position."),
    ("Mengapa sumbu y bar chart wajib dimulai dari 0, sedangkan line chart tidak?",
     "Pada bar chart, nilai dibaca dari **panjang batang**; memotong sumbu membuat selisih tampak berlipat (melanggar *lie factor*). "
     "Line chart dibaca dari **posisi dan kemiringan** (pola naik-turun), jadi sumbu boleh di-zoom ke rentang data."),
    ("Pilih grafik yang paling tepat: (a) sebaran umur pelanggan; (b) tren penjualan 2 bulan; (c) hubungan harga dan qty; "
     "(d) perpindahan pelanggan dari kanal iklan ke cabang; (e) pangsa penjualan per cabang lalu per kategori.",
     "(a) **histogram/boxplot**, (b) **line chart**, (c) **scatter plot**, (d) **sankey diagram** (aliran source → target), "
     "(e) **tree map** (hierarki cabang → kategori, luas = penjualan)."),
    ("Apa itu *redundant encoding*? Beri contoh.",
     "Satu variabel dikodekan di **lebih dari satu channel** agar lebih cepat dan akurat dibaca, misalnya kategori dibedakan dengan "
     "**warna dan bentuk marker** sekaligus (`sns.scatterplot(hue='cabang', style='cabang')`). Ini juga membantu pembaca buta warna."),
    ("Apa pelajaran dari *Anscombe's quartet*?",
     "Empat dataset dengan mean, variansi, korelasi, dan garis regresi yang **hampir identik** ternyata berbentuk sangat berbeda saat diplot "
     "(linear, melengkung, ada outlier, satu titik ekstrem). Jadi **selalu visualisasikan data**, jangan hanya mengandalkan statistik ringkasan."),
]

# =====================================================================================
# EXECUTION (themed SVG capture)
# =====================================================================================
FG, LINE, MUTED, SURF = "#010203", "#040506", "#070809", "#fafbfc"
THEME = {
    "svg.fonttype": "none", "font.size": 9, "text.color": FG, "axes.labelcolor": FG,
    "axes.titlecolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": LINE,
    "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "grid.color": LINE,
}


def themed_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    s = re.sub(r'\s(width|height)="[\d.]+pt"', "", s, count=2)
    for k, v in {FG: "var(--fg)", LINE: "var(--line)", MUTED: "var(--muted)", SURF: "var(--surface)"}.items():
        s = s.replace(k, v)
    s = s.replace("<svg ", '<svg role="img" class="fig out-fig" ', 1)
    return re.sub(r">\s+<", "><", s)


def run(code, ns):
    out = io.StringIO()
    with plt.rc_context(THEME), contextlib.redirect_stdout(out):
        exec(compile(code, "<jawaban>", "exec"), ns)
        figs = [themed_svg(plt.figure(n)) for n in plt.get_fignums()]
    plt.close("all")
    return out.getvalue().rstrip(), figs


ns = {}
setup_out, _ = run(SETUP, ns)
results = []
for s in SOALS:
    out, figs = run(s["jawaban"], ns)
    results.append((out, figs))

# ---- numbers used in explanations ----
df_c, raw = ns["df"], ns["df_raw"]
rk = ns["ringkas"]
med = df_c.groupby("cabang")["durasi_tunggu"].median()
st = ns["stat"]
r = ns["r"]
ar = abs(r)
kekuatan = ("sangat lemah" if ar < .2 else "lemah" if ar < .4 else "sedang" if ar < .6 else "kuat" if ar < .8 else "sangat kuat")
ph = ns["per_hari"]
V = dict(
    n_raw=len(raw), n_bersih=len(df_c), n_dup=int(raw.duplicated().sum()),
    top_cabang=rk.index[0], top_total=rk["total_penjualan"].iloc[0] / 1e6, top_trx=int(rk["jumlah_transaksi"].iloc[0]),
    low_cabang=rk.index[-1], low_total=rk["total_penjualan"].iloc[-1] / 1e6,
    avg_min=rk["rata_per_transaksi"].min(), avg_max=rk["rata_per_transaksi"].max(),
    skew=st["skew"], mean=st["mean"], median=st["median"], n_out=st["n_outlier"],
    med_jkt=f'{med["Jakarta"]:.1f}', med_lain=f'{med.drop("Jakarta").min():.1f}–{med.drop("Jakarta").max():.1f}',
    kopi_min=ns["komp"]["Kopi"].min(), kopi_max=ns["komp"]["Kopi"].max(),
    we=ph.loc[True], wd=ph.loc[False], ratio=ph.loc[True] / ph.loc[False], r=r, kekuatan=kekuatan,
    kopi_top=ns["komp"]["Kopi"].idxmax(), kopi_rest_max=ns["komp"]["Kopi"].drop(ns["komp"]["Kopi"].idxmax()).max(),
    ma_min=ns["ma7"].min() / 1e6, ma_max=ns["ma7"].max() / 1e6,
    we_above=(ns["weekend"] > ns["harian"][ns["harian"].index.dayofweek < 5].median()).mean(),
    r_tq=f'{df_c[["total", "qty"]].corr().iloc[0, 1]:.2f}', r_th=f'{df_c[["total", "harga"]].corr().iloc[0, 1]:.2f}',
)
for s in SOALS:
    s["penjelasan"] = s["penjelasan"].format(**V)
ANALISIS = [(q, qe, a.format(**V)) for q, qe, a in ANALISIS]

# =====================================================================================
# OUTPUT: HTML blocks for the page
# =====================================================================================
import html as H


def mdh(text):
    return md_lib.markdown(text, extensions=["tables"])


def codeblock(code, label):
    return (f'<div class="codeblock"><div class="cb-h"><span>{H.escape(label)}</span>'
            f'<button class="copy" type="button" aria-label="Salin {H.escape(label)}">Salin</button></div>'
            f'<pre><code class="language-python">{H.escape(code)}</code></pre></div>')


def outblock(text):
    return f'<div class="outblock"><div class="cb-h"><span>Output</span></div><pre class="out">{H.escape(text)}</pre></div>' if text else ""


TOPIC = {"cleaning": ("Cleaning", "c-cln"), "categorical": ("Categorical", "c-cat"), "hierarchical": ("Hierarchical", "c-hie"),
         "temporal": ("Temporal", "c-tem"), "relational": ("Relational", "c-rel"), "debug": ("Debugging", "c-dbg")}
sections = []

col_rows = "".join(f"<tr><td><code>{c}</code></td><td>{mdh(k)[3:-4]}</td><td>{t}</td></tr>" for c, k, t in COLUMNS)
total_poin = sum(s["poin"] for s in SOALS)
intro = (f'<div class="pq-intro">{mdh(SCENARIO)}<p class="en">{H.escape(SCENARIO_EN)}</p>'
         f'<div class="pq-meta"><span><b>{len(SOALS)}</b> soal kode · <b>{total_poin}</b> poin</span>'
         f'<span><b>{len(ANALISIS)}</b> tugas analisis</span><span><b>{len(TEORI)}</b> soal teori</span></div>'
         f'<div class="dl"><a href="/notebook/Prediksi_Praktikum_08_Soal.ipynb" download>Unduh notebook soal (.ipynb)</a>'
         f'<a href="/notebook/Prediksi_Praktikum_08_Kunci.ipynb" download>Unduh notebook kunci (.ipynb)</a></div>'
         f'<p class="note">Ini <b>prediksi</b> yang disusun mengikuti gaya soal praktikum IF5100 sebelumnya dan materi Week 11–12. Soal asli bisa berbeda, tetapi konsep yang diuji sama.</p></div>')
sections.append(dict(id="p-intro", title="Studi kasus: Kopi Senja", tag="", intro="Prediksi Praktikum 08 · visualisasi & cleaning data", blocks=[
    dict(title="Skenario", body=intro, search="skenario kopi senja"),
    dict(title="Aturan", body=mdh(RULES), search="aturan"),
    dict(title="Kolom dataset", body=f'<div class="tbl"><table><thead><tr><th>Kolom</th><th>Isi</th><th>Tipe</th></tr></thead><tbody>{col_rows}</tbody></table></div>', search="kolom dataset"),
    dict(title="Setup dataset (jalankan dulu)", body=codeblock(SETUP, "setup.py") + outblock(setup_out), search="setup dataset buat_dataset"),
]))

blocks = []
for s, (out, figs) in zip(SOALS, results):
    tlabel, tcls = TOPIC[s["topik"]]
    figs_html = "".join(f'<figure class="figbox out">{f}</figure>' for f in figs)
    body = (f'<div class="pq-head"><span class="chip {tcls}">{tlabel}</span><span class="pts">{s["poin"]} poin</span></div>'
            f'{mdh(s["teks"])}<p class="en">{H.escape(s["teks_en"])}</p>'
            f'{codeblock(s["kerangka"], "Kerangka kode (kode soal)")}'
            f'<details class="ans"><summary>Lihat jawaban</summary><div class="ans-body">'
            f'{codeblock(s["jawaban"], "Jawaban")}{outblock(out)}{figs_html}'
            f'<div class="why"><h4>Penjelasan</h4>{mdh(s["penjelasan"])}</div></div></details>')
    blocks.append(dict(title=f'Soal {s["no"]} · {s["judul"]}', body=body, search=f'soal {s["no"]} {s["judul"]}'))
sections.append(dict(id="p-soal", title="Soal kode", tag="", intro="Isi fungsi, jalankan kode uji, lalu buat grafiknya. Buka “Lihat jawaban” setelah mencoba sendiri.", blocks=blocks))

qa = "".join(f'<li><div class="q">{mdh(q)[3:-4]}<p class="en">{H.escape(qe)}</p></div>'
             f'<details class="ans"><summary>Lihat jawaban</summary><div class="ans-body">{mdh(a)}</div></details></li>'
             for q, qe, a in ANALISIS)
sections.append(dict(id="p-analisis", title="Tugas analisis", tag="", intro="Dijawab di sel teks berdasarkan hasil Soal 1–6.", blocks=[
    dict(title="Pertanyaan analisis", body=f'<ol class="qa">{qa}</ol>', search="analisis"),
]))
qt = "".join(f'<li><div class="q">{mdh(q)[3:-4]}</div><details class="ans"><summary>Lihat jawaban</summary>'
             f'<div class="ans-body">{mdh(a)}</div></details></li>' for q, a in TEORI)
sections.append(dict(id="p-teori", title="Soal teori", tag="", intro="Konsep dari slide 12a (Data Visualization).", blocks=[
    dict(title="Pertanyaan teori", body=f'<ol class="qa">{qt}</ol>', search="teori"),
]))

(TOOLS / "prediksi.json").write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")

# =====================================================================================
# OUTPUT: notebooks
# =====================================================================================
def nb_cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(kunci):
    cells = [nb_cell("markdown", "# **Prediksi Praktikum 08 - IF5100 Pemrograman untuk Data Analitik**"
                     + (" — KUNCI JAWABAN" if kunci else "") + "\n\nNama : ______________________\n\nNIM  : ______________________"),
             nb_cell("markdown", "### Aturan / Rules\n\n" + RULES),
             nb_cell("markdown", "## Studi kasus: Kopi Senja\n\n" + SCENARIO + "\n\n*" + SCENARIO_EN + "*\n\n| Kolom | Isi | Tipe |\n|---|---|---|\n"
                     + "\n".join(f"| `{c}` | {k} | {t} |" for c, k, t in COLUMNS)),
             nb_cell("markdown", "### Setup dataset\nJalankan sel ini lebih dulu. **Jangan diubah.**"),
             nb_cell("code", SETUP)]
    for s in SOALS:
        cells.append(nb_cell("markdown", f'---\n## SOAL {s["no"]} — {s["judul"]} ({s["poin"]} poin)\n\n{s["teks"]}\n\n*{s["teks_en"]}*'))
        if s["no"] == 7:
            cells.append(nb_cell("code", "# KODE YANG SALAH (untuk dibaca, jangan dijalankan apa adanya)\n" + "\n".join("# " + l if l else "#" for l in s["kerangka"].splitlines())))
        cells.append(nb_cell("code", s["jawaban"] if kunci else (s["kerangka"] if s["no"] != 7 else "# TULIS KODE YANG SUDAH DIPERBAIKI DI SINI\n")))
        cells.append(nb_cell("markdown", ("**Penjelasan:**\n\n" + s["penjelasan"]) if kunci else "**Penjelasan:**\n\n_tulis di sini_"))
    cells.append(nb_cell("markdown", "---\n## Tugas Analisis\nJawab di sel teks di bawah setiap pertanyaan.\n\n*Answer in the text cell below each question.*"))
    for i, (q, qe, a) in enumerate(ANALISIS, 1):
        cells.append(nb_cell("markdown", f"### {i}. {q}\n\n*{qe}*"))
        cells.append(nb_cell("markdown", "**Jawaban:** " + (a if kunci else "")))
    cells.append(nb_cell("markdown", "---\n## Soal Teori"))
    for i, (q, a) in enumerate(TEORI, 1):
        cells.append(nb_cell("markdown", f"### {i}. {q}"))
        cells.append(nb_cell("markdown", "**Jawaban:** " + (a if kunci else "")))
    return {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


NB_DIR.mkdir(parents=True, exist_ok=True)
for kunci, name in [(False, "Prediksi_Praktikum_08_Soal.ipynb"), (True, "Prediksi_Praktikum_08_Kunci.ipynb")]:
    (NB_DIR / name).write_text(json.dumps(notebook(kunci), ensure_ascii=False, indent=1), encoding="utf-8")

print("soal:", len(SOALS), "| analisis:", len(ANALISIS), "| teori:", len(TEORI))
print("figures:", sum(len(f) for _, f in results), "| json KB:", (TOOLS / "prediksi.json").stat().st_size // 1024)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in V.items()})
