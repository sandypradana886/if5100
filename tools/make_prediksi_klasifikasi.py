"""Build the classification practicum prediction notebooks (soal + kunci).

The kunci notebook is executed for real with nbclient (kernel python3), so every output, chart, and number
in the explanations comes from the actual run. Answer texts are templates filled from values that the last
(hidden) cell prints as JSON; that cell is removed afterwards.

Usage (from the project root):
    python tools/make_prediksi_klasifikasi.py
Writes public/notebook/Prediksi_Praktikum_Klasifikasi_{Soal,Kunci}.ipynb.
Needs numpy, pandas, matplotlib, seaborn, scikit-learn, nbformat, nbclient, ipykernel."""
import copy
import json
import sys
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
NB_DIR = ROOT / "public" / "notebook"
BASENAME = "Prediksi_Praktikum_Klasifikasi"


def d(s):
    return textwrap.dedent(s).strip("\n")


CELLS = []   # (role, kind, text): role in both | ans (code answer) | jaw (markdown answer template)


def md(text):
    CELLS.append(("both", "markdown", d(text)))


def code(src):
    CELLS.append(("both", "code", d(src)))


def ans(src, todo="# TODO"):
    CELLS.append(("ans", "code", (d(src), todo)))


def jaw(template):
    CELLS.append(("jaw", "markdown", d(template)))


def soal(huruf, teks, teks_en):
    md(f"**{huruf}.** {d(teks)}\n\n*{d(teks_en)}*")


# =====================================================================================
# HEADER
# =====================================================================================
md("""
# **Prediksi Praktikum Klasifikasi - IF5100 Pemrograman untuk Data Analitik**
## Random Forest, KNN, Regresi Logistik, ANN, dan Ensemble

Nama : ______________________

NIM  : ______________________
""")
md("""
### Aturan / Rules

- Kerjakan di **Google Colab** (runtime CPU). Jalankan sel **berurutan** dari atas ke bawah.
- Library: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. Dataset dibuat atau di-*load* langsung di notebook, tanpa file CSV.
- Gunakan `random_state=42` di **setiap** fungsi yang punya parameter tersebut (`train_test_split`, model, `StratifiedKFold`).
- **Jangan ubah** sel *setup dataset* dan nama variabel yang diminta soal.
- Setiap grafik wajib punya **judul, label sumbu, dan legend** (jika ada lebih dari satu seri).
- Pertanyaan bertanda ✍️ dijawab di sel teks (Markdown) di bawahnya.

*Work in Google Colab, run cells in order, use `random_state=42` everywhere, do not change the setup cells,
and answer the ✍️ questions in the Markdown cell below them.*

| Soal | Topik | Poin |
|---|---|---|
| 1 | Churn pelanggan internet: cleaning, `Pipeline` + `ColumnTransformer`, Regresi Logistik vs KNN vs Random Forest, threshold, interpretasi | 40 |
| 2 | Klasifikasi wine: KNN dengan/tanpa scaling, memilih *k* dengan CV, ANN (`MLPClassifier`), `GridSearchCV` | 25 |
| 3 | Diagnosis kanker payudara: ensemble (Voting, Bagging, Random Forest, Boosting, Stacking) | 25 |
| 4 | Soal teori | 10 |
""")
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.inspection import permutation_importance, DecisionBoundaryDisplay
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
""")

# =====================================================================================
# SOAL 1 — CHURN
# =====================================================================================
md("""
---
## SOAL 1 — Churn Pelanggan Internet Rumah (40 poin)

Penyedia internet rumah **NusaNet** kehilangan banyak pelanggan setiap bulan. Tim retensi ingin model yang dapat
menandai pelanggan yang berisiko **berhenti berlangganan (churn)** agar bisa dihubungi lebih dulu. Data 3.000-an
pelanggan diambil dari dua sistem yang berbeda sehingga masih **kotor**: penulisan kota tidak seragam, ada umur yang
mustahil, ada nilai kosong, dan ada baris ganda.

*NusaNet, a home internet provider, wants a model that flags customers who are likely to churn. The data comes from
two systems and is dirty: inconsistent city names, impossible ages, missing values, and duplicate rows.*

| Kolom | Keterangan |
|---|---|
| `customer_id` | ID pelanggan |
| `gender`, `umur`, `kota` | data demografis (`kota` belum seragam, `umur` ada yang mustahil) |
| `paket` | Basic / Standard / Premium |
| `kontrak` | Bulanan / Tahunan / 2 Tahun |
| `lama_berlangganan` | lama berlangganan (bulan) |
| `biaya_bulanan` | tagihan per bulan (ribu rupiah) |
| `kecepatan_mbps` | kecepatan paket |
| `gangguan_per_bulan` | rata-rata gangguan jaringan per bulan |
| `tiket_komplain` | jumlah tiket komplain 3 bulan terakhir |
| `metode_bayar` | Auto-debit / Transfer / E-wallet |
| `churn` | **target**: Ya / Tidak |

### Setup dataset (jangan diubah)
""")
code("""
def buat_data_churn(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    kontrak = rng.choice(["Bulanan", "Tahunan", "2 Tahun"], n, p=[0.5, 0.3, 0.2])
    paket = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])
    lama = np.where(kontrak == "Bulanan", rng.integers(1, 36, n), rng.integers(6, 73, n))
    biaya = np.select([paket == "Basic", paket == "Standard"], [250, 400], 650) + rng.normal(0, 40, n)
    kecepatan = np.select([paket == "Basic", paket == "Standard"], [20, 50], 100)
    gangguan = rng.poisson(1.2, n)
    komplain = rng.poisson(0.3 + 0.4 * gangguan)
    metode = rng.choice(["Auto-debit", "Transfer", "E-wallet"], n, p=[0.35, 0.35, 0.30])
    z = (-2.0 + 1.6 * (kontrak == "Bulanan") + 0.3 * (kontrak == "Tahunan") - 0.035 * lama
         + 0.35 * gangguan + 0.45 * komplain + 0.002 * (biaya - 400) - 0.6 * (metode == "Auto-debit")
         + rng.normal(0, 0.5, n))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Ya", "Tidak")
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "gender": rng.choice(["Laki-laki", "Perempuan"], n),
        "umur": rng.integers(18, 70, n).astype(float),
        "kota": rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan"], n),
        "paket": paket, "kontrak": kontrak, "lama_berlangganan": lama,
        "biaya_bulanan": biaya.round(1), "kecepatan_mbps": kecepatan,
        "gangguan_per_bulan": gangguan, "tiket_komplain": komplain,
        "metode_bayar": metode, "churn": churn})
    # --- data kotor ---
    i = rng.choice(n, 450, replace=False)
    df.loc[i[:150], "kota"] = df.loc[i[:150], "kota"].str.upper()
    df.loc[i[150:300], "kota"] = " " + df.loc[i[150:300], "kota"].str.lower() + " "
    bdg = df.index[df["kota"] == "Bandung"][:60]
    df.loc[bdg, "kota"] = "Bdg"
    df.loc[rng.choice(n, 25, replace=False), "umur"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "umur"] = rng.choice([0, 150, 999], 8)
    df.loc[rng.choice(n, 40, replace=False), "biaya_bulanan"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "metode_bayar"] = np.nan
    df = pd.concat([df, df.sample(30, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


df_churn = buat_data_churn()
""")

soal("a", """
Tampilkan jumlah baris dan kolom `df_churn`, informasi tipe data (`info()`), dan 5 baris pertama. Lalu tampilkan
jumlah serta persentase pelanggan `churn` = Ya dan Tidak, beserta perbandingannya dalam **pie chart**.
✍️ Apakah kelasnya seimbang? Metrik apa yang sebaiknya diperhatikan selain accuracy?
""", """
Show the shape, info(), and the first 5 rows. Then show the count and percentage of each churn class with a pie chart.
✍️ Are the classes balanced? Which metric should be watched besides accuracy?
""")
ans("""
print("Jumlah baris, kolom:", df_churn.shape)
df_churn.info()
display(df_churn.head())

jumlah = df_churn["churn"].value_counts()
print(pd.DataFrame({"jumlah": jumlah, "persen": (jumlah / len(df_churn) * 100).round(2)}))

plt.figure(figsize=(4.5, 4.5))
plt.pie(jumlah, labels=jumlah.index, autopct="%.1f%%", colors=["#4c72b0", "#dd8452"], startangle=90)
plt.title("Proporsi pelanggan churn")
plt.show()
""")
jaw("""
Kelas **tidak seimbang**: churn = Ya hanya {churn_pct:.1f}% ({churn_n} dari {n_raw} baris). Model yang selalu menebak
"Tidak" saja sudah mendapat accuracy ±{baseline:.1f}%, padahal tidak menemukan satu pun pelanggan churn. Karena itu
yang diperhatikan adalah **recall** kelas churn (berapa banyak pelanggan churn yang tertangkap), **precision**, **F1**,
dan **ROC-AUC**.
""")

soal("b", """
Lakukan *data cleaning* dan simpan hasilnya di `df1`:
1. Tampilkan jumlah baris duplikat, lalu hapus.
2. Hapus kolom `customer_id`.
3. Seragamkan `kota` (hapus spasi, huruf kapital di awal kata, `Bdg` → `Bandung`), lalu tampilkan `value_counts()`.
4. Ubah `umur` yang mustahil (< 17 atau > 90) menjadi `NaN`.
5. Ubah `churn` menjadi biner (Ya = 1, Tidak = 0).
6. Tampilkan jumlah missing value per kolom. **Jangan isi** missing value di sini (akan diisi di dalam `Pipeline`).

✍️ Mengapa pengisian missing value sebaiknya dilakukan di dalam `Pipeline` setelah split, bukan sebelum split?
""", """
Clean the data into df1: drop duplicates, drop customer_id, standardize kota, set impossible ages to NaN, encode
churn as 1/0, and show missing values per column (do not impute yet).
✍️ Why should imputation happen inside the Pipeline after the split?
""")
ans("""
df1 = df_churn.copy()
print("Duplikat:", df1.duplicated().sum())
df1 = df1.drop_duplicates().drop(columns="customer_id")

df1["kota"] = df1["kota"].str.strip().str.title().replace({"Bdg": "Bandung"})
print(df1["kota"].value_counts(), "\\n")

umur_salah = (df1["umur"] < 17) | (df1["umur"] > 90)
print("Umur mustahil:", umur_salah.sum())
df1.loc[umur_salah, "umur"] = np.nan

df1["churn"] = df1["churn"].map({"Ya": 1, "Tidak": 0})
print("\\nMissing value per kolom:")
print(df1.isna().sum())
print("\\nUkuran df1:", df1.shape)
""")
jaw("""
Terdapat {dup} baris duplikat dan {umur_salah} umur mustahil; setelah dibersihkan `df1` berisi {n1} baris. Kota kini
hanya 4 nilai. Missing value tersisa di `umur` ({na_umur}), `biaya_bulanan` ({na_biaya}), dan `metode_bayar` ({na_metode}).

Median/modus untuk mengisi NaN adalah **statistik dari data**. Jika dihitung dari seluruh data sebelum split, informasi
dari data test ikut masuk ke proses latih (**data leakage**) sehingga skor test terlalu optimis. Di dalam `Pipeline`,
`SimpleImputer` hanya di-*fit* pada data train (dan pada fold train saat cross-validation).
""")

soal("c", """
Eksplorasi faktor churn:
1. Buat 1 figure 1×3: **tingkat churn (%) per `kontrak`** (bar), **tingkat churn (%) per `metode_bayar`** (bar), dan
   **boxplot `lama_berlangganan` per kelas churn**.
2. Tampilkan korelasi setiap kolom numerik dengan `churn`, diurutkan.

✍️ Sebutkan 3 temuan terpenting.
""", """
Plot the churn rate per contract type and per payment method, and a boxplot of tenure per churn class (1×3 figure).
Show the correlation of each numeric column with churn. ✍️ Write the 3 most important findings.
""")
ans("""
rate_kontrak = (df1.groupby("kontrak")["churn"].mean() * 100).sort_values(ascending=False)
rate_bayar = (df1.groupby("metode_bayar")["churn"].mean() * 100).sort_values(ascending=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
rate_kontrak.plot.bar(ax=ax[0], color="#dd8452", rot=0)
ax[0].set(title="Tingkat churn per kontrak", xlabel="Kontrak", ylabel="Churn (%)")
rate_bayar.plot.bar(ax=ax[1], color="#4c72b0", rot=0)
ax[1].set(title="Tingkat churn per metode bayar", xlabel="Metode bayar", ylabel="Churn (%)")
sns.boxplot(data=df1, x="churn", y="lama_berlangganan", hue="churn", ax=ax[2], legend=False)
ax[2].set(title="Lama berlangganan per kelas churn", xlabel="Churn (0 = Tidak, 1 = Ya)", ylabel="Bulan")
plt.tight_layout()
plt.show()

kor_churn = df1.select_dtypes("number").corr()["churn"].drop("churn").sort_values()
print(rate_kontrak.round(1), "\\n")
print(rate_bayar.round(1), "\\n")
print(kor_churn.round(3))
""")
jaw("""
1. **Kontrak** adalah pembeda terkuat: churn pelanggan Bulanan {rate_bulanan:.1f}%, jauh di atas Tahunan
   ({rate_tahunan:.1f}%) dan 2 Tahun ({rate_2th:.1f}%).
2. **Lama berlangganan** berkorelasi negatif dengan churn (r = {kor_lama:.3f}): pelanggan baru paling mudah pergi,
   terlihat dari median lama berlangganan yang lebih rendah pada kelas churn di boxplot.
3. **Kualitas layanan**: `tiket_komplain` (r = {kor_komplain:.3f}) dan `gangguan_per_bulan` (r = {kor_gangguan:.3f})
   berkorelasi positif. Pembayaran **Auto-debit** memiliki churn terendah ({rate_auto:.1f}% vs {rate_bayar_max:.1f}%
   pada {bayar_max}).
""")

soal("d", """
Siapkan data untuk model:
1. Pisahkan `X` (semua kolom kecuali `churn`) dan `y`. Bagi **80:20** dengan `stratify=y` dan `random_state=42`.
   Tampilkan ukuran dan proporsi churn di train dan test.
2. Buat daftar `num_cols` dan `cat_cols`.
3. Buat fungsi `buat_preprocessor(scale=True)` yang mengembalikan `ColumnTransformer` **baru**: numerik →
   `SimpleImputer(median)` (+ `StandardScaler` jika `scale=True`); kategorikal → `SimpleImputer(most_frequent)` +
   `OneHotEncoder(handle_unknown="ignore")`.
""", """
Split 80:20 with stratify, list numeric and categorical columns, and write buat_preprocessor(scale=True) that returns
a new ColumnTransformer (median imputer + optional scaler for numeric; mode imputer + one-hot for categorical).
""")
ans("""
X = df1.drop(columns="churn")
y = df1["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", X_train.shape, "| test:", X_test.shape)
print(f"proporsi churn  train = {y_train.mean():.3f} | test = {y_test.mean():.3f}")

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("numerik   :", num_cols)
print("kategorik :", cat_cols)


def buat_preprocessor(scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")),
                 ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols),
                              ("cat", Pipeline(cat_steps), cat_cols)],
                             verbose_feature_names_out=False)
""")

soal("e", """
Latih tiga model dalam `Pipeline` (`buat_preprocessor()` + model) dan simpan di dict `models_1`:
- `LogisticRegression(max_iter=1000)`
- `KNeighborsClassifier(n_neighbors=15)`
- `RandomForestClassifier(n_estimators=300, random_state=42)` (preprocessor tanpa scaling)

Hitung **accuracy, precision, recall, F1, dan ROC-AUC** pada data test, simpan dalam DataFrame `hasil_1`
(index = nama model). ✍️ Model mana yang paling cocok untuk tim retensi dan mengapa?
""", """
Train Logistic Regression, KNN (k=15), and Random Forest pipelines, store them in models_1, and build hasil_1 with
accuracy, precision, recall, F1, and ROC-AUC on the test set. ✍️ Which model suits the retention team best, and why?
""")
ans("""
models_1 = {
    "Logistic Regression": Pipeline([("prep", buat_preprocessor()),
                                     ("model", LogisticRegression(max_iter=1000))]),
    "KNN (k=15)": Pipeline([("prep", buat_preprocessor()),
                            ("model", KNeighborsClassifier(n_neighbors=15))]),
    "Random Forest": Pipeline([("prep", buat_preprocessor(scale=False)),
                               ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
}
rows = []
for nama, model in models_1.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({"model": nama,
                 "accuracy": accuracy_score(y_test, pred),
                 "precision": precision_score(y_test, pred),
                 "recall": recall_score(y_test, pred),
                 "f1": f1_score(y_test, pred),
                 "roc_auc": roc_auc_score(y_test, proba)})
hasil_1 = pd.DataFrame(rows).set_index("model").round(4)
display(hasil_1)
""")
jaw("""
{best1_text}

Untuk tim retensi, kesalahan yang paling mahal adalah **false negative** (pelanggan yang akan churn tidak terdeteksi),
jadi recall dan F1 lebih penting daripada accuracy. Selisih accuracy antar model kecil, tetapi ROC-AUC dan F1
menunjukkan seberapa baik model memisahkan kedua kelas.
""")

soal("f", """
1. Tampilkan **confusion matrix** Regresi Logistik dalam bentuk heatmap (label `Tidak`/`Ya`).
2. Turunkan **threshold** Regresi Logistik dari 0.5 menjadi 0.3 (`predict_proba(X_test)[:, 1] >= 0.3`).
   Bandingkan precision, recall, F1, dan jumlah pelanggan yang ditandai churn untuk kedua threshold dalam DataFrame
   `hasil_threshold`.

✍️ Jelaskan *trade-off* yang terjadi dan threshold mana yang Anda sarankan.
""", """
Plot the Logistic Regression confusion matrix as a heatmap, then compare thresholds 0.5 and 0.3 (precision, recall,
F1, number of flagged customers) in hasil_threshold. ✍️ Explain the trade-off and recommend a threshold.
""")
ans("""
lr_churn = models_1["Logistic Regression"]
proba_lr = lr_churn.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, (proba_lr >= 0.5).astype(int))
plt.figure(figsize=(4.5, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
plt.title("Confusion matrix Regresi Logistik (threshold 0.5)")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

rows = []
for t in [0.5, 0.3]:
    pred_t = (proba_lr >= t).astype(int)
    rows.append({"threshold": t, "precision": precision_score(y_test, pred_t),
                 "recall": recall_score(y_test, pred_t), "f1": f1_score(y_test, pred_t),
                 "ditandai_churn": int(pred_t.sum())})
hasil_threshold = pd.DataFrame(rows).set_index("threshold").round(4)
display(hasil_threshold)
""")
jaw("""
Pada threshold 0.5, dari {pos_test} pelanggan churn di data test, model menangkap {tp} (TP) dan melewatkan {fn} (FN);
ada {fp} pelanggan setia yang salah ditandai (FP).

Menurunkan threshold ke 0.3 membuat model lebih mudah menyebut "churn": recall naik dari {rec05:.3f} ke {rec03:.3f},
tetapi precision turun dari {prec05:.3f} ke {prec03:.3f} dan jumlah pelanggan yang ditandai naik dari {flag05} ke
{flag03}. {thr_saran}
""")

soal("g", """
Interpretasi model:
1. Ambil nama fitur hasil preprocessing (`get_feature_names_out()`), lalu tampilkan **5 koefisien Regresi Logistik**
   dengan nilai absolut terbesar.
2. Tampilkan **5 feature importance** teratas Random Forest.
3. Buat 1 figure 1×2 berisi kedua bar chart tersebut.

✍️ Jelaskan arti tanda koefisien dan apakah kedua model sepakat tentang fitur terpenting.
""", """
Show the 5 Logistic Regression coefficients with the largest absolute values and the top 5 Random Forest feature
importances, as two bar charts in one 1×2 figure. ✍️ Explain the sign of the coefficients and whether both models agree.
""")
ans("""
fitur = lr_churn.named_steps["prep"].get_feature_names_out()
koef = pd.Series(lr_churn.named_steps["model"].coef_[0], index=fitur)
top_koef = koef.reindex(koef.abs().sort_values(ascending=False).index).head(5)

rf_churn = models_1["Random Forest"]
imp = pd.Series(rf_churn.named_steps["model"].feature_importances_,
                index=rf_churn.named_steps["prep"].get_feature_names_out())
top_imp = imp.sort_values(ascending=False).head(5)
print(top_koef.round(3), "\\n")
print(top_imp.round(3))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
top_koef[::-1].plot.barh(ax=ax[0], color=np.where(top_koef[::-1] > 0, "#c44e52", "#55a868"))
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set(title="5 koefisien terbesar (Regresi Logistik)", xlabel="Koefisien (fitur ter-scale)", ylabel="Fitur")
top_imp[::-1].plot.barh(ax=ax[1], color="#4c72b0")
ax[1].set(title="5 feature importance teratas (Random Forest)", xlabel="Importance (MDI)", ylabel="Fitur")
plt.tight_layout()
plt.show()
""")
jaw("""
Koefisien Regresi Logistik dihitung pada fitur yang sudah di-scale (numerik) atau one-hot (kategorikal). **Positif**
berarti menaikkan peluang churn, **negatif** menurunkan. Lima koefisien terbesar: {koef_text}.
Contohnya, `{koef1}` = {koef1_val:+.3f}, artinya {koef1_arti}.

Lima fitur terpenting Random Forest: {imp_text}. Kedua model sepakat bahwa {sepakat}. Perbedaannya,
importance RF tidak punya tanda (hanya seberapa sering dan seberapa baik fitur dipakai membagi data) dan cenderung
lebih tinggi untuk fitur numerik kontinu dengan banyak nilai unik. {umur_text}
""")

soal("h", """
**(BONUS)** Latih ulang Regresi Logistik dengan `class_weight="balanced"`. Bandingkan precision, recall, dan F1-nya
dengan Regresi Logistik biasa. ✍️ Apa efek `class_weight="balanced"`?
""", """
(BONUS) Retrain Logistic Regression with class_weight="balanced" and compare precision, recall, and F1 with the
default model. ✍️ What does class_weight="balanced" do?
""")
ans("""
lr_bal = Pipeline([("prep", buat_preprocessor()),
                   ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(X_train, y_train)
banding = {}
for nama, m in [("LogReg biasa", lr_churn), ("LogReg balanced", lr_bal)]:
    p = m.predict(X_test)
    banding[nama] = {"precision": precision_score(y_test, p), "recall": recall_score(y_test, p),
                     "f1": f1_score(y_test, p)}
hasil_balanced = pd.DataFrame(banding).T.round(4)
display(hasil_balanced)
""")
jaw("""
`class_weight="balanced"` memberi bobot kesalahan pada kelas minoritas sebanding dengan kelangkaannya
(bobot = n_sampel / (n_kelas × n_sampel_kelas)). Model menjadi lebih "berani" menebak churn: recall naik dari
{bal_rec0:.3f} ke {bal_rec1:.3f}, precision turun dari {bal_prec0:.3f} ke {bal_prec1:.3f}, dan F1 berubah dari
{bal_f10:.3f} ke {bal_f11:.3f}. Efeknya mirip menurunkan threshold, tetapi terjadi saat training.
""")

# =====================================================================================
# SOAL 2 — WINE (KNN & ANN)
# =====================================================================================
md("""
---
## SOAL 2 — Klasifikasi Wine dengan KNN dan ANN (25 poin)

Dataset bawaan scikit-learn **wine** berisi 13 hasil analisis kimia dari 178 wine yang berasal dari 3 kultivar
(`class_0`, `class_1`, `class_2`). Skala fiturnya sangat berbeda, misalnya `proline` ratusan sampai ribuan,
sedangkan `hue` di bawah 2.

*The built-in wine dataset has 13 chemical measurements of 178 wines from 3 cultivars. Feature scales differ a lot.*

### Setup dataset (jangan diubah)
""")
code("""
from sklearn.datasets import load_wine
wine = load_wine(as_frame=True)
""")
soal("a", """
Buat DataFrame `df_wine` dari `wine.frame`, tambahkan kolom `kelas` berisi nama kelas (`wine.target_names`).
Tampilkan ukuran data, jumlah data per kelas, serta nilai **min, max, mean, dan std** setiap fitur.
✍️ Fitur mana yang skalanya paling besar dan paling kecil? Apa akibatnya untuk KNN?
""", """
Create df_wine with a kelas column of class names. Show the shape, class counts, and min/max/mean/std of each feature.
✍️ Which features have the largest and smallest scale, and what does that mean for KNN?
""")
ans("""
df_wine = wine.frame.copy()
df_wine["kelas"] = df_wine["target"].map(dict(enumerate(wine.target_names)))
print("Ukuran:", df_wine.shape)
print(df_wine["kelas"].value_counts().sort_index(), "\\n")
skala = df_wine[wine.feature_names].describe().T[["min", "max", "mean", "std"]].sort_values("std", ascending=False)
display(skala.round(3))
""")
jaw("""
`proline` memiliki skala terbesar (rentang {pro_min:.0f}–{pro_max:.0f}, std {pro_std:.1f}), sedangkan `{fit_kecil}` paling
kecil (std {kecil_std:.3f}). KNN menghitung **jarak Euclidean**, sehingga tanpa scaling jarak hampir seluruhnya
ditentukan oleh `proline` (dan `magnesium`); 11 fitur lain praktis diabaikan.
""")

soal("b", """
Pisahkan `X_w` (13 fitur) dan `y_w` (`target`), lalu bagi **75:25** dengan `stratify` dan `random_state=42`.
Tampilkan ukuran train dan test.
""", """
Split X_w and y_w 75:25 with stratify and random_state=42, and show the shapes.
""")
ans("""
X_w = df_wine[wine.feature_names]
y_w = df_wine["target"]
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_w, y_w, test_size=0.25, stratify=y_w, random_state=42)
print("train:", Xw_train.shape, "| test:", Xw_test.shape)
""")

soal("c", """
Latih `KNeighborsClassifier(n_neighbors=5)` **tanpa scaling** dan **dengan** `StandardScaler` (dalam `Pipeline`).
Tampilkan accuracy test keduanya. ✍️ Jelaskan perbedaannya.
""", """
Train KNN (k=5) without scaling and with StandardScaler in a Pipeline, and show both test accuracies. ✍️ Explain.
""")
ans("""
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(Xw_train, yw_train)
knn_sc = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]).fit(Xw_train, yw_train)
acc_knn_raw = accuracy_score(yw_test, knn_raw.predict(Xw_test))
acc_knn_sc = accuracy_score(yw_test, knn_sc.predict(Xw_test))
print(f"KNN tanpa scaling : {acc_knn_raw:.4f}")
print(f"KNN dengan scaling: {acc_knn_sc:.4f}")
""")
jaw("""
Accuracy naik dari **{acc_knn_raw:.4f}** menjadi **{acc_knn_sc:.4f}** setelah scaling. Tanpa scaling, tetangga
"terdekat" hanyalah wine dengan kadar `proline` yang mirip. Setelah `StandardScaler` (mean 0, std 1), setiap fitur
berkontribusi seimbang pada jarak sehingga fitur pembeda lain (`flavanoids`, `color_intensity`, `alcohol`) ikut dipakai.
""")

soal("d", """
Cari *k* terbaik untuk KNN dengan scaling: untuk *k* = 1, 3, 5, …, 29 hitung rata-rata accuracy
**5-fold `StratifiedKFold`** (`shuffle=True`, `random_state=42`) **pada data train**. Simpan di Series `cv_k`,
buat line plot *k* vs accuracy CV, lalu latih ulang model dengan *k* terbaik (`best_k`) dan tampilkan accuracy test.
Jika beberapa *k* memiliki skor CV yang sama, pilih *k* **terbesar** (model lebih stabil).
✍️ Mengapa *k* dipilih dengan CV di data train, bukan dengan data test?
""", """
For k = 1, 3, …, 29, compute the mean 5-fold stratified CV accuracy on the training data (with scaling), plot it,
retrain with the best k (largest k on ties), and show the test accuracy. ✍️ Why choose k with CV on train instead of the test set?
""")
ans("""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ks = range(1, 30, 2)
cv_k = pd.Series({k: cross_val_score(Pipeline([("sc", StandardScaler()),
                                               ("knn", KNeighborsClassifier(n_neighbors=k))]),
                                     Xw_train, yw_train, cv=skf).mean() for k in ks})
best_k = int(cv_k[np.isclose(cv_k, cv_k.max())].index.max())   # seri → k terbesar

plt.figure(figsize=(7, 4))
plt.plot(cv_k.index, cv_k.values, marker="o", label="accuracy CV (5-fold)")
plt.axvline(best_k, color="red", ls="--", label=f"k terbaik = {best_k}")
plt.title("KNN (dengan scaling): pengaruh k")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

knn_best = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k))]).fit(Xw_train, yw_train)
acc_knn_best = accuracy_score(yw_test, knn_best.predict(Xw_test))
print("best_k =", best_k, "| CV =", round(cv_k.max(), 4), "| test =", round(acc_knn_best, 4))
""")
jaw("""
*k* terbaik = **{best_k}** dengan accuracy CV {cv_best:.4f} dan accuracy test {acc_knn_best:.4f}. {k_text}

Jika *k* dipilih berdasarkan skor test, data test ikut menentukan model sehingga skornya tidak lagi menggambarkan
performa pada data baru. CV di data train memberi estimasi yang jujur, dan data test hanya dipakai sekali di akhir.
""")

soal("e", """
Buat **ANN** dengan `Pipeline` `StandardScaler` → `MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
random_state=42)` dan simpan di `ann_wine`. Tampilkan accuracy train dan test, jumlah epoch (`n_iter_`), lalu plot
**kurva loss** (`loss_curve_`).
""", """
Build an ANN pipeline (StandardScaler → MLPClassifier((32, 16), max_iter=2000, random_state=42)) as ann_wine.
Show train/test accuracy and the number of epochs, and plot the loss curve.
""")
ans("""
ann_wine = Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42))])
ann_wine.fit(Xw_train, yw_train)
mlp = ann_wine.named_steps["mlp"]
acc_ann_train = ann_wine.score(Xw_train, yw_train)
acc_ann_test = ann_wine.score(Xw_test, yw_test)
print(f"train = {acc_ann_train:.4f} | test = {acc_ann_test:.4f} | epoch = {mlp.n_iter_}")

plt.figure(figsize=(7, 4))
plt.plot(mlp.loss_curve_, label="loss train")
plt.title("Kurva loss MLPClassifier (wine)")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.show()
""")

soal("f", """
Lakukan `GridSearchCV` (cv = `skf`, scoring accuracy) pada pipeline ANN dengan grid:
`hidden_layer_sizes` ∈ {(16,), (32, 16), (64, 32)}, `alpha` ∈ {0.0001, 0.01, 1}, `learning_rate_init` ∈ {0.001, 0.01}.
Tampilkan parameter terbaik, skor CV terbaik, dan accuracy test model terbaik (`ann_best`).
✍️ Apa fungsi parameter `alpha`?
""", """
Run GridSearchCV on the ANN pipeline with the given grid. Show the best parameters, best CV score, and the test
accuracy of the best model (ann_best). ✍️ What does alpha do?
""")
ans("""
grid_ann = GridSearchCV(
    Pipeline([("sc", StandardScaler()), ("mlp", MLPClassifier(max_iter=2000, random_state=42))]),
    param_grid={"mlp__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
                "mlp__alpha": [0.0001, 0.01, 1],
                "mlp__learning_rate_init": [0.001, 0.01]},
    cv=skf, scoring="accuracy", n_jobs=-1)
grid_ann.fit(Xw_train, yw_train)
ann_best = grid_ann.best_estimator_
acc_ann_best = accuracy_score(yw_test, ann_best.predict(Xw_test))
print("parameter terbaik:", grid_ann.best_params_)
print(f"CV terbaik = {grid_ann.best_score_:.4f} | test = {acc_ann_best:.4f}")
""")
jaw("""
Parameter terbaik: `{ann_params}` dengan accuracy CV {ann_cv:.4f} dan accuracy test {acc_ann_best:.4f}.
`alpha` adalah kekuatan **regularisasi L2**: makin besar, bobot jaringan makin ditekan agar tidak terlalu besar
sehingga model lebih sederhana dan tidak mudah overfit. Pada dataset kecil seperti wine ({wine_train_n} data latih), regularisasi
membantu generalisasi.
""")

soal("g", """
Buat DataFrame `hasil_2` berisi accuracy test untuk: KNN tanpa scaling, KNN dengan scaling (k=5), KNN *k* terbaik,
ANN (32, 16), dan ANN hasil GridSearch. Tampilkan juga `classification_report` model terbaik.
✍️ Model mana yang Anda pilih? Mengapa scaling penting untuk KNN **dan** ANN?
""", """
Build hasil_2 with the test accuracy of the five models and show the classification report of the best one.
✍️ Which model do you choose? Why is scaling important for both KNN and ANN?
""")
ans("""
hasil_2 = pd.Series({"KNN tanpa scaling": acc_knn_raw, "KNN scaling (k=5)": acc_knn_sc,
                     f"KNN scaling (k={best_k})": acc_knn_best, "ANN (32, 16)": acc_ann_test,
                     "ANN GridSearch": acc_ann_best}, name="accuracy_test").sort_values(ascending=False).round(4)
display(hasil_2.to_frame())
model_wine = {"KNN tanpa scaling": knn_raw, "KNN scaling (k=5)": knn_sc, f"KNN scaling (k={best_k})": knn_best,
              "ANN (32, 16)": ann_wine, "ANN GridSearch": ann_best}
terbaik_2 = hasil_2.index[0]
print("Model terbaik:", terbaik_2)
print(classification_report(yw_test, model_wine[terbaik_2].predict(Xw_test), target_names=wine.target_names, digits=3))
""")
jaw("""
{wine_text}

KNN memakai jarak sehingga fitur berskala besar mendominasi. ANN dilatih dengan *gradient descent*: jika skala input
berbeda jauh, gradien untuk tiap bobot juga berbeda jauh sehingga training lambat atau tidak stabil (konvergensi buruk).
Karena itu keduanya wajib memakai `StandardScaler`, sedangkan model berbasis pohon tidak memerlukannya.
""")

soal("h", """
**(BONUS)** Dengan dua fitur saja, `alcohol` dan `proline`, gambarkan **decision boundary** KNN (*k* = 15) tanpa scaling
dan dengan scaling dalam 1 figure 1×2 (`DecisionBoundaryDisplay`). ✍️ Apa yang terlihat?
""", """
(BONUS) Using only alcohol and proline, draw the KNN (k=15) decision boundary without and with scaling (1×2 figure).
✍️ What do you see?
""")
ans("""
dua = ["alcohol", "proline"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, (judul, model) in zip(ax, [("tanpa scaling", KNeighborsClassifier(n_neighbors=15)),
                                  ("dengan scaling", Pipeline([("sc", StandardScaler()),
                                                               ("knn", KNeighborsClassifier(n_neighbors=15))]))]):
    model.fit(Xw_train[dua], yw_train)
    DecisionBoundaryDisplay.from_estimator(model, Xw_train[dua], response_method="predict", alpha=0.3, ax=a)
    sns.scatterplot(data=Xw_train[dua].assign(kelas=wine.target_names[yw_train]), x="alcohol", y="proline",
                    hue="kelas", palette="tab10", ax=a, s=30)
    a.set(title=f"KNN k=15 {judul} (acc test = {model.score(Xw_test[dua], yw_test):.3f})",
          xlabel="alcohol", ylabel="proline")
plt.tight_layout()
plt.show()
""")
jaw("""
Tanpa scaling, batas keputusan hampir **horizontal**: hanya `proline` yang menentukan kelas karena selisih `alcohol`
(±1) tidak berarti dibanding selisih `proline` (ratusan). Setelah scaling, batasnya memanfaatkan kedua sumbu sehingga
kelas `class_1` (alcohol rendah) bisa dipisahkan dari `class_2`.
""")

# =====================================================================================
# SOAL 3 — BREAST CANCER (ENSEMBLE)
# =====================================================================================
md("""
---
## SOAL 3 — Diagnosis Kanker Payudara dengan Ensemble (25 poin)

Dataset bawaan **breast cancer** berisi 30 fitur hasil citra sel dari 569 pasien, dengan diagnosis *malignant*
(ganas) atau *benign* (jinak). Pada dataset ini `target = 0` berarti **malignant**. Kesalahan paling berbahaya adalah
pasien ganas yang diprediksi jinak (**false negative**).

*The built-in breast cancer dataset has 30 features from 569 patients. target = 0 means malignant. The most dangerous
error is a malignant tumour predicted as benign (false negative).*

### Setup dataset (jangan diubah)
""")
code("""
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
""")
soal("a", """
Buat `df_bc` dari `bc.frame`, lalu buat kolom `ganas` = 1 jika `target == 0` (malignant) dan 0 jika sebaliknya.
Tampilkan jumlah dan persentase setiap kelas `ganas`. ✍️ Mengapa kelas positif (1) sebaiknya adalah *malignant*?
""", """
Create df_bc and a column ganas = 1 for malignant (target == 0). Show the count and percentage of each class.
✍️ Why should malignant be the positive class (1)?
""")
ans("""
df_bc = bc.frame.copy()
df_bc["ganas"] = (df_bc["target"] == 0).astype(int)
jumlah_bc = df_bc["ganas"].value_counts()
print(pd.DataFrame({"jumlah": jumlah_bc, "persen": (jumlah_bc / len(df_bc) * 100).round(2)}))
""")
jaw("""
Ada {bc_ganas} kasus ganas ({bc_ganas_pct:.1f}%) dan {bc_jinak} jinak. `precision_score`, `recall_score`, dan
`f1_score` secara default menghitung untuk kelas **1**. Dengan malignant = 1, *recall* langsung berarti "berapa persen
pasien kanker yang terdeteksi", yaitu metrik yang paling penting secara medis.
""")

soal("b", """
Pisahkan `X_b` (30 fitur) dan `y_b` (`ganas`), bagi **80:20** dengan `stratify` dan `random_state=42`.
Buat fungsi `metrik(model)` yang mengembalikan dict **accuracy, precision, recall, dan F1** model pada data test.
""", """
Split X_b and y_b 80:20 with stratify and random_state=42. Write metrik(model) that returns a dict of test accuracy,
precision, recall, and F1.
""")
ans("""
X_b = df_bc[bc.feature_names]
y_b = df_bc["ganas"]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_b, y_b, test_size=0.2, stratify=y_b, random_state=42)
print("train:", Xb_train.shape, "| test:", Xb_test.shape)


def metrik(model):
    pred = model.predict(Xb_test)
    return {"accuracy": accuracy_score(yb_test, pred), "precision": precision_score(yb_test, pred),
            "recall": recall_score(yb_test, pred), "f1": f1_score(yb_test, pred)}
""")

soal("c", """
Buat tiga **model dasar** dan simpan di list `base`:
- `("lr", Pipeline(StandardScaler, LogisticRegression(max_iter=1000)))`
- `("knn", Pipeline(StandardScaler, KNeighborsClassifier(n_neighbors=7)))`
- `("tree", DecisionTreeClassifier(max_depth=4, random_state=42))`

Untuk tiap model, hitung rata-rata **accuracy dan recall 5-fold CV** di data train (`cross_validate`, cv = `skf`),
simpan di DataFrame `cv_base`.
""", """
Create the three base models in the list base and compute their mean 5-fold CV accuracy and recall on the training
data with cross_validate, stored in cv_base.
""")
ans("""
base = [
    ("lr", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])),
    ("knn", Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=7))])),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
]
cv_base = pd.DataFrame({
    nama: {k: v.mean() for k, v in cross_validate(m, Xb_train, yb_train, cv=skf,
                                                  scoring=["accuracy", "recall"]).items() if k.startswith("test_")}
    for nama, m in base}).T.round(4)
display(cv_base)
""")

soal("d", """
Gabungkan ketiga model dasar dengan `VotingClassifier`, versi **hard** (`vote_hard`) dan **soft** (`vote_soft`).
Latih di data train dan tampilkan `metrik()` keduanya beserta ketiga model dasar dalam satu DataFrame `hasil_vote`.
✍️ Apa beda hard dan soft voting?
""", """
Combine the base models with VotingClassifier (hard and soft), train them, and show metrik() for the two voting models
and the three base models in hasil_vote. ✍️ What is the difference between hard and soft voting?
""")
ans("""
vote_hard = VotingClassifier(base, voting="hard").fit(Xb_train, yb_train)
vote_soft = VotingClassifier(base, voting="soft").fit(Xb_train, yb_train)
fitted_base = {nama: m.fit(Xb_train, yb_train) for nama, m in base}
hasil_vote = pd.DataFrame({**{nama: metrik(m) for nama, m in fitted_base.items()},
                           "Voting hard": metrik(vote_hard), "Voting soft": metrik(vote_soft)}).T.round(4)
display(hasil_vote)
""")
jaw("""
**Hard voting** mengambil label terbanyak dari ketiga model. **Soft voting** merata-ratakan `predict_proba` lalu
memilih kelas dengan peluang tertinggi, sehingga keyakinan model ikut diperhitungkan (model yang sangat yakin bisa
mengalahkan dua model yang ragu). Hasil test: Voting hard accuracy {vh_acc:.4f} / recall {vh_rec:.4f}, Voting soft
accuracy {vs_acc:.4f} / recall {vs_rec:.4f}, dibanding model dasar terbaik `{base_best}` (accuracy {base_best_acc:.4f}).
""")

soal("e", """
Bandingkan **Decision Tree tunggal** (`max_depth=None`, `random_state=42`) dengan ensemble berbasis pohon:
- `BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=42)`
- `RandomForestClassifier(n_estimators=300, random_state=42)`
- `AdaBoostClassifier(n_estimators=200, random_state=42)`
- `GradientBoostingClassifier(random_state=42)`

Tampilkan accuracy **train** dan `metrik()` test dalam DataFrame `hasil_tree`. ✍️ Jelaskan perbedaan bagging dan
boosting berdasarkan hasilnya.
""", """
Compare a single fully grown Decision Tree with Bagging, Random Forest, AdaBoost, and Gradient Boosting. Show the train
accuracy and the test metrics in hasil_tree. ✍️ Explain bagging vs boosting using the results.
""")
ans("""
model_tree = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
rows = {}
for nama, m in model_tree.items():
    m.fit(Xb_train, yb_train)
    rows[nama] = {"acc_train": m.score(Xb_train, yb_train), **metrik(m)}
hasil_tree = pd.DataFrame(rows).T.round(4)
display(hasil_tree)
""")
jaw("""
Decision Tree tunggal mencapai accuracy train {dt_train:.4f} tetapi test {dt_test:.4f} (overfit, variance tinggi).
**Bagging** dan **Random Forest** melatih banyak pohon secara paralel pada *bootstrap sample* lalu voting; rata-rata
banyak pohon menurunkan variance sehingga accuracy test naik menjadi {bag_test:.4f} dan {rf_test:.4f}. Random Forest
juga mengacak subset fitur di tiap split sehingga pohon-pohonnya lebih beragam.
**Boosting** (AdaBoost {ada_test:.4f}, Gradient Boosting {gb_test:.4f}) melatih model lemah secara **berurutan**; setiap
model baru fokus pada kesalahan model sebelumnya sehingga yang diturunkan terutama **bias**.
""")

soal("f", """
Buat `StackingClassifier` dengan `estimators = base + [("rf", RandomForestClassifier(n_estimators=300,
random_state=42))]`, `final_estimator=LogisticRegression(max_iter=1000)`, dan `cv=5`. Tampilkan `metrik()`-nya.
✍️ Mengapa stacking melatih *meta-model* dari prediksi *out-of-fold*?
""", """
Build a StackingClassifier (base models + Random Forest, final LogisticRegression, cv=5) and show its metrics.
✍️ Why does stacking train the meta-model on out-of-fold predictions?
""")
ans("""
stack = StackingClassifier(
    estimators=base + [("rf", RandomForestClassifier(n_estimators=300, random_state=42))],
    final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1)
stack.fit(Xb_train, yb_train)
hasil_stack = pd.Series(metrik(stack)).round(4)
print(hasil_stack)
""")
jaw("""
Stacking: accuracy {st_acc:.4f}, recall {st_rec:.4f}, F1 {st_f1:.4f}. Jika meta-model dilatih dari prediksi model dasar
pada data yang sama dengan yang dipakai melatihnya, prediksi itu terlalu bagus (model dasar sudah "hafal"), sehingga
meta-model belajar terlalu percaya pada model yang overfit. Prediksi *out-of-fold* (setiap fold diprediksi oleh model
yang tidak melihat fold itu) meniru kondisi data baru.
""")

soal("g", """
Gabungkan semua hasil Soal 3 (model dasar, voting, model pohon, stacking) ke DataFrame `hasil_3`, urutkan berdasarkan
**recall** lalu F1, dan buat **bar chart** recall dan F1 per model. ✍️ Model mana yang Anda rekomendasikan untuk skrining
kanker dan mengapa?
""", """
Combine all Soal 3 results in hasil_3, sort by recall then F1, and plot recall and F1 per model.
✍️ Which model do you recommend for cancer screening, and why?
""")
ans("""
hasil_3 = pd.concat([hasil_vote, hasil_tree.drop(columns="acc_train"),
                     hasil_stack.to_frame("Stacking").T]).sort_values(["recall", "f1"], ascending=False)
display(hasil_3)

hasil_3[["recall", "f1"]].plot.barh(figsize=(8, 5.5), xlim=(0.8, 1.0))
plt.gca().invert_yaxis()
plt.title("Recall dan F1 (kelas ganas) per model")
plt.xlabel("Skor")
plt.ylabel("Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
""")
jaw("""
{rekom_text}

Pada skrining kanker, **recall** kelas ganas lebih penting daripada accuracy: false negative berarti pasien kanker
dipulangkan tanpa penanganan, sedangkan false positive "hanya" berujung pemeriksaan lanjutan. Dengan data test {bc_test_n}
pasien ({bc_test_pos} ganas), selisih satu pasien = {bc_poin:.1f} poin recall, jadi selisih kecil antar model sebaiknya dikonfirmasi dengan CV.
""")

soal("h", """
**(BONUS)** Tampilkan 10 fitur terpenting Random Forest berdasarkan `feature_importances_` dan berdasarkan
**permutation importance** pada data test (`n_repeats=10`, `random_state=42`) dalam 1 figure 1×2.
✍️ Mengapa hasil keduanya bisa berbeda?
""", """
(BONUS) Show the top 10 Random Forest features by feature_importances_ and by permutation importance on the test set
(1×2 figure). ✍️ Why can the two rankings differ?
""")
ans("""
rf_bc = model_tree["Random Forest"]
mdi = pd.Series(rf_bc.feature_importances_, index=bc.feature_names).sort_values(ascending=False).head(10)
perm = permutation_importance(rf_bc, Xb_test, yb_test, n_repeats=10, random_state=42, n_jobs=-1)
perm = pd.Series(perm.importances_mean, index=bc.feature_names).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
mdi[::-1].plot.barh(ax=ax[0], color="#4c72b0")
ax[0].set(title="Feature importance (MDI)", xlabel="Importance", ylabel="Fitur")
perm[::-1].plot.barh(ax=ax[1], color="#dd8452")
ax[1].set(title="Permutation importance (data test)", xlabel="Penurunan accuracy", ylabel="Fitur")
plt.tight_layout()
plt.show()
""")
jaw("""
Teratas menurut MDI: {mdi_top}; menurut permutation: {perm_top}. **MDI** dihitung dari data train (seberapa besar
fitur menurunkan Gini) dan membagi importance di antara fitur yang saling berkorelasi (misalnya `radius`, `perimeter`,
`area`). **Permutation importance** mengukur turunnya accuracy di data test saat satu kolom diacak; jika ada fitur lain
yang membawa informasi serupa, mengacak satu fitur hampir tidak berdampak sehingga nilainya kecil.
""")

# =====================================================================================
# TEORI
# =====================================================================================
TEORI = [
    ("Jelaskan perbedaan **bagging** dan **boosting**, beserta contoh model masing-masing.",
     "Bagging melatih banyak model **paralel** pada bootstrap sample lalu menggabungkan dengan voting/rata-rata; tujuannya "
     "menurunkan **variance** (contoh: `BaggingClassifier`, `RandomForestClassifier`). Boosting melatih model lemah "
     "**berurutan**, tiap model fokus pada kesalahan sebelumnya; tujuannya menurunkan **bias** (contoh: "
     "`AdaBoostClassifier`, `GradientBoostingClassifier`, XGBoost)."),
    ("Mengapa `StandardScaler` dan `SimpleImputer` harus di-*fit* hanya pada data train?",
     "Mean, std, dan median adalah informasi dari data. Jika dihitung dengan data test, informasi test bocor ke "
     "training (**data leakage**) sehingga skor evaluasi terlalu optimis. Data test harus diperlakukan seperti data "
     "baru yang belum pernah dilihat; cukup `transform`, tidak `fit`."),
    ("Apa fungsi `stratify=y` pada `train_test_split`?",
     "Menjaga proporsi setiap kelas di data train dan test sama dengan data asli. Tanpa stratify, pada data tidak "
     "seimbang test bisa berisi terlalu sedikit (atau terlalu banyak) kelas minoritas sehingga evaluasi tidak stabil."),
    ("Apa pengaruh nilai *k* yang terlalu kecil dan terlalu besar pada KNN?",
     "*k* kecil (mis. 1): model mengikuti titik terdekat, sensitif noise → **overfitting** (variance tinggi). *k* besar: "
     "voting melibatkan banyak titik jauh, batas keputusan terlalu halus → **underfitting** (bias tinggi). *k* dipilih "
     "dengan cross-validation."),
    ("Mengapa Random Forest tidak memerlukan scaling, sedangkan KNN, Regresi Logistik, dan ANN memerlukannya?",
     "Pohon keputusan membagi data dengan ambang per fitur (`x <= t`); scaling tidak mengubah urutan nilai sehingga "
     "split yang dihasilkan sama. KNN memakai jarak (fitur berskala besar mendominasi), sedangkan Regresi Logistik "
     "dan ANN memakai optimasi gradien dan regularisasi yang sensitif terhadap skala."),
    ("Kapan accuracy menyesatkan? Beri contoh kapan *recall* lebih penting dan kapan *precision* lebih penting.",
     "Saat kelas tidak seimbang: model yang selalu menebak kelas mayoritas bisa ber-accuracy tinggi tanpa menemukan "
     "kelas minoritas. **Recall** penting jika false negative mahal (skrining kanker, deteksi fraud, churn). "
     "**Precision** penting jika false positive mahal (filter spam yang membuang email penting, vonis bersalah)."),
    ("Jelaskan peran fungsi aktivasi, `alpha`, dan `early_stopping` pada `MLPClassifier`.",
     "Fungsi aktivasi (mis. ReLU) membuat jaringan mampu memodelkan hubungan **non-linier**; tanpa aktivasi, banyak "
     "layer setara dengan satu model linier. `alpha` adalah regularisasi L2 yang menekan bobot agar tidak overfit. "
     "`early_stopping=True` menyisihkan sebagian data train sebagai validasi dan menghentikan training ketika skor "
     "validasi berhenti membaik."),
    ("Apa beda *hard voting* dan *soft voting*, dan apa syarat soft voting?",
     "Hard voting memilih label terbanyak; soft voting merata-ratakan peluang (`predict_proba`) dan memilih kelas "
     "dengan peluang rata-rata tertinggi. Soft voting mensyaratkan semua model dasar punya `predict_proba` dan "
     "biasanya lebih akurat karena memakai tingkat keyakinan model."),
    ("Apa itu *OOB score* pada Random Forest?",
     "Setiap pohon dilatih pada bootstrap sample, sehingga ±37% data tidak terpilih (*out-of-bag*) untuk pohon itu. "
     "Data OOB diprediksi oleh pohon-pohon yang tidak melihatnya, menghasilkan estimasi akurasi seperti validasi "
     "tanpa perlu memisahkan data (`oob_score=True`)."),
    ("Mengapa hasil koefisien Regresi Logistik baru bisa dibandingkan antar fitur setelah scaling?",
     "Besar koefisien bergantung pada satuan fitur: koefisien per 1 rupiah pasti jauh lebih kecil daripada per 1 "
     "tahun. Setelah standarisasi, koefisien berarti perubahan log-odds per kenaikan **1 standar deviasi**, sehingga "
     "besarnya bisa dibandingkan untuk melihat fitur mana yang paling berpengaruh."),
]
md("""
---
## SOAL 4 — Teori (10 poin)

Jawab singkat di sel teks di bawah setiap pertanyaan. *Answer briefly in the text cell below each question.*
""")
for i, (q, a) in enumerate(TEORI, 1):
    md(f"**{i}.** {q}")
    CELLS.append(("teori", "markdown", a))

# =====================================================================================
# VALUES FOR ANSWER TEMPLATES (hidden cell, removed after execution)
# =====================================================================================
VALUES_CELL = r'''
import json as _json
_V = {}
_vc = df_churn["churn"].value_counts()
_V.update(n_raw=len(df_churn), churn_n=int(_vc["Ya"]), churn_pct=_vc["Ya"] / len(df_churn) * 100,
          baseline=_vc["Tidak"] / len(df_churn) * 100,
          dup=int(df_churn.duplicated().sum()), umur_salah=int(umur_salah.sum()), n1=len(df1),
          na_umur=int(df1["umur"].isna().sum()), na_biaya=int(df1["biaya_bulanan"].isna().sum()),
          na_metode=int(df1["metode_bayar"].isna().sum()),
          rate_bulanan=rate_kontrak["Bulanan"], rate_tahunan=rate_kontrak["Tahunan"], rate_2th=rate_kontrak["2 Tahun"],
          kor_lama=kor_churn["lama_berlangganan"], kor_komplain=kor_churn["tiket_komplain"],
          kor_gangguan=kor_churn["gangguan_per_bulan"], rate_auto=rate_bayar["Auto-debit"],
          rate_bayar_max=rate_bayar.iloc[0], bayar_max=rate_bayar.index[0])
_V["hasil_1"] = hasil_1.reset_index().to_dict("records")
_tn, _fp, _fn, _tp = confusion_matrix(y_test, (proba_lr >= 0.5).astype(int)).ravel()
_V.update(tp=int(_tp), fn=int(_fn), fp=int(_fp), pos_test=int(y_test.sum()),
          rec05=hasil_threshold.loc[0.5, "recall"], rec03=hasil_threshold.loc[0.3, "recall"],
          prec05=hasil_threshold.loc[0.5, "precision"], prec03=hasil_threshold.loc[0.3, "precision"],
          f105=hasil_threshold.loc[0.5, "f1"], f103=hasil_threshold.loc[0.3, "f1"],
          flag05=int(hasil_threshold.loc[0.5, "ditandai_churn"]), flag03=int(hasil_threshold.loc[0.3, "ditandai_churn"]))
_V["top_koef"] = {k: float(v) for k, v in top_koef.items()}
_V["top_imp"] = {k: float(v) for k, v in top_imp.items()}
_V.update(bal_rec0=hasil_balanced.loc["LogReg biasa", "recall"], bal_rec1=hasil_balanced.loc["LogReg balanced", "recall"],
          bal_prec0=hasil_balanced.loc["LogReg biasa", "precision"], bal_prec1=hasil_balanced.loc["LogReg balanced", "precision"],
          bal_f10=hasil_balanced.loc["LogReg biasa", "f1"], bal_f11=hasil_balanced.loc["LogReg balanced", "f1"])
_V.update(pro_min=skala.loc["proline", "min"], pro_max=skala.loc["proline", "max"], pro_std=skala.loc["proline", "std"],
          fit_kecil=skala.index[-1], kecil_std=skala["std"].iloc[-1],
          acc_knn_raw=acc_knn_raw, acc_knn_sc=acc_knn_sc, best_k=best_k, cv_best=cv_k.max(), cv_k1=cv_k[1],
          acc_knn_best=acc_knn_best, acc_ann_train=acc_ann_train, acc_ann_test=acc_ann_test, ann_epoch=int(mlp.n_iter_),
          ann_params=str(grid_ann.best_params_), ann_cv=grid_ann.best_score_, acc_ann_best=acc_ann_best)
_V["hasil_2"] = {k: float(v) for k, v in hasil_2.items()}
_V.update(bc_test_n=len(yb_test), bc_test_pos=int(yb_test.sum()), bc_poin=100 / yb_test.sum(),
          wine_train_n=len(yw_train), wine_test_n=len(yw_test))
_V.update(bc_ganas=int(jumlah_bc[1]), bc_jinak=int(jumlah_bc[0]), bc_ganas_pct=jumlah_bc[1] / len(df_bc) * 100)
_V["cv_base"] = cv_base.to_dict("index")
_V["hasil_vote"] = hasil_vote.to_dict("index")
_V["hasil_tree"] = hasil_tree.to_dict("index")
_V["hasil_stack"] = hasil_stack.to_dict()
_V["hasil_3"] = hasil_3.to_dict("index")
_V["hasil_3_order"] = hasil_3.index.tolist()
_V["mdi_top"] = mdi.index[:3].tolist()
_V["perm_top"] = perm.index[:3].tolist()
print("__VALUES__" + _json.dumps(_V, default=float))
'''


def fmt_list(names, vals=None):
    if vals is None:
        return ", ".join(f"`{n}`" for n in names)
    return ", ".join(f"`{n}` ({v:+.3f})" for n, v in zip(names, vals))


def derive(V):
    """Derived texts that depend on the actual ranking of results."""
    h1 = {r["model"]: r for r in V["hasil_1"]}
    best_f1 = max(h1, key=lambda m: h1[m]["f1"])
    best_auc = max(h1, key=lambda m: h1[m]["roc_auc"])
    rows = "; ".join(f"{m}: accuracy {r['accuracy']:.4f}, recall {r['recall']:.4f}, F1 {r['f1']:.4f}, AUC {r['roc_auc']:.4f}"
                     for m, r in h1.items())
    V["best1_text"] = (f"Hasil test — {rows}. **{best_f1}** memiliki F1 tertinggi"
                       + (f" sekaligus ROC-AUC tertinggi" if best_f1 == best_auc else f", sedangkan ROC-AUC tertinggi milik {best_auc}")
                       + f". Saya memilih **{best_f1}**"
                       + (": selain skornya terbaik, koefisiennya bisa dijelaskan ke tim bisnis dan peluang churn-nya bisa dipakai untuk mengatur threshold."
                          if best_f1 == "Logistic Regression" else ": skornya paling seimbang antara precision dan recall.")
                       + " Hubungan fitur dengan churn pada data ini cenderung monoton (makin lama berlangganan makin kecil peluang churn, makin banyak komplain makin besar), sehingga model linier sudah cukup; Random Forest dan KNN tidak mendapat keuntungan dari fleksibilitasnya.")
    V["thr_saran"] = ("Karena biaya kehilangan pelanggan biasanya jauh lebih besar daripada biaya menghubungi pelanggan yang "
                      f"sebenarnya setia, threshold **0.3** lebih cocok untuk tim retensi (F1 {V['f105']:.3f} → {V['f103']:.3f})."
                      if V["f103"] >= V["f105"] - 0.02 else
                      f"F1 turun dari {V['f105']:.3f} ke {V['f103']:.3f}; threshold 0.3 hanya disarankan jika kapasitas tim retensi cukup untuk menghubungi lebih banyak pelanggan.")
    names, vals = list(V["top_koef"]), list(V["top_koef"].values())
    V["koef_text"] = fmt_list(names, vals)
    V["koef1"], V["koef1_val"] = names[0], vals[0]
    arti = {
        "kontrak_Bulanan": "pelanggan berkontrak bulanan jauh lebih berisiko churn dibanding kontrak lain",
        "kontrak_2 Tahun": "kontrak 2 tahun menurunkan peluang churn",
        "lama_berlangganan": "setiap kenaikan 1 std lama berlangganan menurunkan log-odds churn",
        "tiket_komplain": "setiap kenaikan 1 std jumlah komplain menaikkan log-odds churn",
    }
    V["koef1_arti"] = arti.get(names[0], ("fitur ini menaikkan" if vals[0] > 0 else "fitur ini menurunkan") + " peluang churn")
    V["imp_text"] = fmt_list(list(V["top_imp"]))
    common = [n for n in V["top_imp"] if n in V["top_koef"]]
    V["sepakat"] = (fmt_list(common) + " termasuk fitur terpenting") if common else "fitur terpentingnya berbeda"
    V["umur_text"] = ("Contohnya `umur`: masuk 5 besar importance RF, padahal tidak masuk 5 besar koefisien Regresi "
                      "Logistik dan tingkat churn hampir sama di semua umur. Ini bias MDI terhadap fitur kontinu, "
                      "bukan bukti bahwa umur memengaruhi churn."
                      if "umur" in V["top_imp"] and "umur" not in V["top_koef"] else "")
    V["k_text"] = (f"Skor *k* = 1 sama tinggi ({V['cv_k1']:.4f}), tetapi *k* = 1 hanya mengandalkan satu tetangga sehingga "
                   "sangat sensitif terhadap noise (variance tinggi); karena itu saat seri dipilih *k* yang lebih besar. "
                   if abs(V["cv_k1"] - V["cv_best"]) < 1e-9 and V["best_k"] != 1 else
                   f"*k* = 1 memberi CV {V['cv_k1']:.4f}: model terlalu mengikuti titik tunggal (variance tinggi). ") +                   "*k* yang terlalu besar membuat batas keputusan terlalu halus (bias tinggi)."
    h2 = V["hasil_2"]
    top2 = list(h2)[0]
    tied = [m for m, v in h2.items() if abs(v - h2[top2]) < 1e-9]
    V["wine_text"] = (f"Accuracy test: " + ", ".join(f"{m} {v:.4f}" for m, v in h2.items()) + ". "
                      + (f"Model terbaik **{top2}**." if len(tied) == 1 else
                         f"{' dan '.join(tied)} sama-sama terbaik ({h2[top2]:.4f}); dengan hanya {V['wine_test_n']} data test, selisih 1 sampel = {100 / V['wine_test_n']:.1f} poin, jadi saya memilih yang lebih sederhana dan cepat, yaitu KNN dengan scaling.")
                      + f" KNN tanpa scaling jauh tertinggal ({h2['KNN tanpa scaling']:.4f}).")
    second = [m for m in h2 if m not in tied][0]
    if len(tied) == 1 and h2[top2] - h2[second] <= 1 / V["wine_test_n"] + 1e-9:
        V["wine_text"] += (f" Namun selisih {top2} dengan {second} hanya 1 dari {V['wine_test_n']} data test, jadi keduanya "
                           "praktis setara; skor CV di data train lebih bisa dipegang untuk membandingkan keduanya.")
    hv = V["hasil_vote"]
    V.update(vh_acc=hv["Voting hard"]["accuracy"], vh_rec=hv["Voting hard"]["recall"],
             vs_acc=hv["Voting soft"]["accuracy"], vs_rec=hv["Voting soft"]["recall"])
    base_best = max(["lr", "knn", "tree"], key=lambda m: hv[m]["accuracy"])
    V.update(base_best=base_best, base_best_acc=hv[base_best]["accuracy"])
    ht = V["hasil_tree"]
    V.update(dt_train=ht["Decision Tree"]["acc_train"], dt_test=ht["Decision Tree"]["accuracy"],
             bag_test=ht["Bagging"]["accuracy"], rf_test=ht["Random Forest"]["accuracy"],
             ada_test=ht["AdaBoost"]["accuracy"], gb_test=ht["Gradient Boosting"]["accuracy"])
    st = V["hasil_stack"]
    V.update(st_acc=st["accuracy"], st_rec=st["recall"], st_f1=st["f1"])
    h3, order = V["hasil_3"], V["hasil_3_order"]
    top = order[0]
    V["rekom_text"] = (f"Urutan teratas berdasarkan recall: " + ", ".join(
        f"{m} (recall {h3[m]['recall']:.4f}, F1 {h3[m]['f1']:.4f})" for m in order[:4])
        + f". Saya merekomendasikan **{top}** karena recall kelas ganasnya paling tinggi"
        + f" (hanya {round((1 - h3[top]['recall']) * V['bc_test_pos'])} dari {V['bc_test_pos']} pasien ganas di data test yang terlewat)"
        + f" dengan precision {h3[top]['precision']:.4f} yang tetap tinggi.")
    V["mdi_top"] = fmt_list(V["mdi_top"])
    V["perm_top"] = fmt_list(V["perm_top"])
    return V


# =====================================================================================
# BUILD
# =====================================================================================
def make_nb(kunci, outputs=None, V=None):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    cells = []
    for role, kind, text in CELLS:
        if role == "both":
            if kind == "markdown" and text.startswith("# **Prediksi") and kunci:
                text = text.replace("Data Analitik**", "Data Analitik** — KUNCI JAWABAN", 1)
            cells.append(nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text))
        elif role == "ans":
            src, todo = text
            cells.append(nbformat.v4.new_code_cell(src if kunci else todo))
        elif role == "jaw":
            body = text.format(**V) if (kunci and V) else ("_tulis jawaban di sini_" if not kunci else text)
            cells.append(nbformat.v4.new_markdown_cell("✍️ **Jawaban:**\n\n" + body))
        elif role == "teori":
            cells.append(nbformat.v4.new_markdown_cell("**Jawaban:** " + (text if kunci else "_tulis jawaban di sini_")))
    nb.cells = cells
    return nb


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = make_nb(kunci=True)
    nb.cells.append(nbformat.v4.new_code_cell(VALUES_CELL))
    print("executing kunci notebook ...", flush=True)
    NotebookClient(nb, kernel_name="python3", timeout=900, resources={"metadata": {"path": str(TOOLS)}}).execute()
    raw = "".join(o.get("text", "") for o in nb.cells[-1].outputs if o.get("output_type") == "stream")
    V = derive(json.loads(raw.split("__VALUES__", 1)[1]))
    errors = [c for c in nb.cells if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.outputs)]
    if errors:
        for c in errors:
            print("ERROR in cell:\n", c.source[:300], "\n", [o for o in c.outputs if o.output_type == "error"][0]["evalue"])
        sys.exit(1)
    # fill templates into the executed notebook: rebuild markdown, keep executed code cells' outputs
    filled = make_nb(kunci=True, V=V)
    executed_code = [c for c in nb.cells[:-1] if c.cell_type == "code"]
    it = iter(executed_code)
    for i, c in enumerate(filled.cells):
        if c.cell_type == "code":
            filled.cells[i] = next(it)
    for i, c in enumerate(filled.cells):
        if c.cell_type == "code":
            c.execution_count = i + 1
            for o in c.outputs:
                if "execution_count" in o:
                    o["execution_count"] = i + 1
    filled.metadata = nb.metadata if "language_info" in nb.metadata else filled.metadata
    filled.metadata["colab"] = {"provenance": []}
    nbformat.write(filled, NB_DIR / f"{BASENAME}_Kunci.ipynb")
    nbformat.write(make_nb(kunci=False), NB_DIR / f"{BASENAME}_Soal.ipynb")
    (TOOLS / "prediksi_klasifikasi_values.json").write_text(
        json.dumps({k: v for k, v in V.items()}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("written:", f"{BASENAME}_Soal.ipynb", f"{BASENAME}_Kunci.ipynb")
    for k in ["churn_pct", "best1_text", "thr_saran", "koef_text", "imp_text", "wine_text", "rekom_text"]:
        print(f"- {k}: {V[k]}")
    print(json.dumps({k: V[k] for k in ["hasil_1", "hasil_2", "hasil_vote", "hasil_tree", "hasil_stack"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
