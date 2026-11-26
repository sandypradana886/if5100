"""Content for the "Klasifikasi" tab: preprocessing → model (Logistic Regression, KNN, Random Forest,
ANN, Ensemble) → evaluation → analysis. Imported by tools/build_page.py.

Same format as tools/content_p06_p09.py: tab dict(id, label, flow, fixture, sections=[...]);
section dict(id, title, desc, cards=[...]); card dict(t, code | html, note, wide, test).

Running example: mobile_prices.csv (tugas Classification Classroom Study), loaded from GitHub.
Run `python tools/content_klasifikasi.py` to execute every code card in order (needs internet for the
CSV, plus numpy, pandas, matplotlib, seaborn, scikit-learn; the Keras card also needs tensorflow).
Set QUICK=1 to skip the slow cards (Keras, stacking, randomized search)."""

URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/mobile_prices.csv"

KLASIFIKASI = dict(
    id="klasifikasi", label="Klasifikasi",
    flow=["EDA", "Preprocessing", "Split stratify", "Pipeline + model", "Tuning (CV di train)", "Evaluasi test", "Analisis"],
    fixture=r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
""",
    sections=[
        # ------------------------------------------------------------------ alur
        dict(id="k-alur", title="Alur kerja & import", desc="Urutan yang dinilai di tugas klasifikasi: data uji hanya disentuh sekali, di akhir.", cards=[
            dict(t="Peta alur klasifikasi", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>#</th><th>Langkah</th><th>Isi</th><th>Output yang ditulis di notebook</th></tr></thead><tbody>
<tr><td>1</td><td>EDA</td><td>shape, tipe, missing, duplikat, distribusi kelas, nilai mencurigakan, hubungan fitur–target</td><td>1–2 kalimat: kelas seimbang? fitur apa yang paling berpengaruh?</td></tr>
<tr><td>2</td><td>Preprocessing</td><td>nilai tidak masuk akal → NaN, imputasi, encoding, scaling, feature engineering/selection</td><td>alasan tiap keputusan</td></tr>
<tr><td>3</td><td>Split</td><td><code>train_test_split(test_size=0.2, stratify=y, random_state=42)</code></td><td>ukuran train/test, proporsi kelas tetap sama</td></tr>
<tr><td>4</td><td>Model</td><td><code>Pipeline([('prep', ...), ('model', ...)])</code> untuk LogReg, KNN, RF, ANN, Ensemble</td><td>hyperparameter yang dipakai</td></tr>
<tr><td>5</td><td>Tuning</td><td><code>GridSearchCV</code> / <code>cross_val_score</code> <b>di data train</b></td><td>parameter terbaik + skor CV</td></tr>
<tr><td>6</td><td>Evaluasi</td><td>accuracy, confusion matrix, classification report (+ F1 macro, ROC-AUC)</td><td>tabel perbandingan semua model</td></tr>
<tr><td>7</td><td>Analisis</td><td>model terbaik &amp; alasannya, overfitting, kelas yang tertukar, fitur penting</td><td>paragraf 3–5 kalimat (Markdown)</td></tr>
</tbody></table></div>"""),
            dict(t="Import lengkap", wide=True, code=r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, cross_validate,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance, DecisionBoundaryDisplay
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay)

RANDOM_STATE = 42
sns.set_theme(style='whitegrid')"""),
            dict(t="Lima model sekilas", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Model</th><th>Cara kerja</th><th>Perlu scaling?</th><th>Kelebihan</th><th>Kekurangan</th><th>Hyperparameter utama</th></tr></thead><tbody>
<tr><td><b>Regresi Logistik</b></td><td>kombinasi linier fitur → sigmoid/softmax → peluang kelas</td><td><b>Ya</b> (konvergensi &amp; regularisasi adil)</td><td>cepat, koefisien bisa ditafsirkan, peluang terkalibrasi</td><td>batas keputusan linier</td><td><code>C</code>, <code>l1_ratio</code>, <code>max_iter</code>, <code>class_weight</code></td></tr>
<tr><td><b>KNN</b></td><td>cari k tetangga terdekat, voting mayoritas</td><td><b>Wajib</b> (berbasis jarak)</td><td>tanpa asumsi bentuk data, mudah dijelaskan</td><td>lambat saat prediksi, sensitif fitur tak relevan &amp; dimensi tinggi</td><td><code>n_neighbors</code>, <code>weights</code>, <code>p</code>/<code>metric</code></td></tr>
<tr><td><b>Random Forest</b></td><td>banyak decision tree dari bootstrap + subset fitur acak, voting</td><td>Tidak</td><td>non-linier, tahan outlier, ada feature importance, minim tuning</td><td>mudah hafal train (acc train ≈ 1), model besar, sulit ditafsirkan</td><td><code>n_estimators</code>, <code>max_depth</code>, <code>min_samples_leaf</code>, <code>max_features</code></td></tr>
<tr><td><b>ANN (MLP)</b></td><td>lapisan neuron + aktivasi, dilatih backpropagation</td><td><b>Wajib</b></td><td>bisa memodelkan pola sangat kompleks</td><td>butuh data &amp; tuning, hasil bergantung seed, black box</td><td><code>hidden_layer_sizes</code>, <code>alpha</code>, <code>learning_rate_init</code>, <code>early_stopping</code></td></tr>
<tr><td><b>Ensemble</b></td><td>gabungkan beberapa model (voting, stacking, bagging, boosting)</td><td>ikut model dasarnya</td><td>biasanya paling stabil &amp; akurat</td><td>lebih lambat, lebih sulit dijelaskan</td><td><code>voting</code>, <code>weights</code>, <code>final_estimator</code>, <code>learning_rate</code></td></tr>
</tbody></table></div>"""),
        ]),
        # ------------------------------------------------------------------ EDA
        dict(id="k-eda", title="EDA untuk klasifikasi", desc="Fokus: kualitas data, keseimbangan kelas, dan fitur mana yang membedakan kelas.", cards=[
            dict(t="Load & cek awal", code=r"""
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/mobile_prices.csv"
df = pd.read_csv(url)                      # (2000, 21)
df.shape; df.head(); df.info()
df.isna().sum().sum()                      # total missing
df.duplicated().sum()                      # baris duplikat
df.describe().T                            # min = 0 yang aneh? max ekstrem?
df.nunique().sort_values()                 # nunique == 2 → fitur biner"""),
            dict(t="Distribusi kelas (imbalance?)", code=r"""
TARGET = 'price_range'
print(df[TARGET].value_counts().sort_index())
print(df[TARGET].value_counts(normalize=True).round(3))
rasio = df[TARGET].value_counts().max() / df[TARGET].value_counts().min()
print('rasio mayoritas/minoritas:', round(rasio, 2))   # > 1.5 → mulai perhatikan imbalance
sns.countplot(data=df, x=TARGET); plt.title('Jumlah per kelas'); plt.show()""",
                 note="Seimbang (di sini 500 per kelas) → accuracy layak dipakai. Tidak seimbang → utamakan F1 macro, recall per kelas, <code>class_weight='balanced'</code>."),
            dict(t="Nilai mencurigakan", code=r"""
# fitur yang secara fisik tidak mungkin 0
cek_nol = ['sc_w', 'px_height', 'sc_h', 'px_width', 'mobile_wt', 'battery_power']
print((df[cek_nol] == 0).sum())            # sc_w = 0 → 180 baris, px_height = 0 → 2 baris
# tidak masuk akal secara logika
print('px_height > px_width :', (df['px_height'] > df['px_width']).sum())
print('three_g = 0 tapi four_g = 1 :', ((df['three_g'] == 0) & (df['four_g'] == 1)).sum())"""),
            dict(t="Fitur vs target", code=r"""
fitur = df.drop(columns=TARGET).columns
kor = df.corr()[TARGET].drop(TARGET).sort_values()
kor.plot.barh(figsize=(6, 6)); plt.title('Korelasi fitur dengan price_range'); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col in zip(ax, ['ram', 'battery_power', 'px_width']):
    sns.boxplot(data=df, x=TARGET, y=col, ax=a); a.set_title(col)
plt.tight_layout(); plt.show()

pd.crosstab(df['four_g'], df[TARGET], normalize='columns').round(2)   # fitur biner vs kelas""",
                 note="Di dataset ini <code>ram</code> berkorelasi 0.92 dengan target; <code>battery_power</code>, <code>px_width</code>, <code>px_height</code> menyusul (0.15–0.20). Sisanya hampir 0."),
            dict(t="Korelasi antar fitur (multikolinearitas)", code=r"""
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False)
plt.title('Matriks korelasi'); plt.show()
c = df.drop(columns=TARGET).corr().abs()
pasangan = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack().sort_values(ascending=False)
pasangan.head(5)          # pc–fc, three_g–four_g, px_height–px_width, sc_h–sc_w""",
                 note="Korelasi tinggi antar fitur mengganggu tafsir koefisien regresi logistik, tetapi tidak banyak berpengaruh pada RF/KNN."),
            dict(t="Checklist temuan EDA", html=r"""
<ul>
<li>Ukuran data, jumlah fitur numerik / biner / kategorikal</li>
<li>Missing &amp; duplikat (jumlah dan keputusan)</li>
<li>Distribusi kelas: seimbang atau tidak → metrik yang dipakai</li>
<li>Nilai mustahil (0 pada ukuran layar, umur negatif) → jadikan NaN</li>
<li>Skala fitur sangat berbeda (ram ribuan vs m_dep &lt; 1) → scaling untuk KNN, LogReg, ANN</li>
<li>Fitur paling membedakan kelas (korelasi, boxplot per kelas)</li>
<li>Pasangan fitur yang sangat berkorelasi</li>
</ul>"""),
        ]),
        # ------------------------------------------------------------------ preprocessing
        dict(id="k-prep", title="Preprocessing & feature engineering", desc="Transformasi yang belajar dari data (imputer, scaler, selector) dimasukkan ke Pipeline agar hanya di-fit pada data latih.", cards=[
            dict(t="Nilai 0 yang tidak masuk akal → NaN", code=r"""
df_clean = df.copy()
for col in ['sc_w', 'px_height']:
    df_clean[col] = df_clean[col].replace(0, np.nan)
df_clean.isna().sum()[lambda s: s > 0]     # sc_w 180, px_height 2
# imputasi TIDAK dilakukan di sini, tapi di Pipeline (SimpleImputer) setelah split""",
                 note="Membuang 180 baris = buang 9% data. Mengganti NaN dengan median <b>di dalam Pipeline</b> lebih aman dari leakage."),
            dict(t="Feature engineering", code=r"""
df_fe = df_clean.copy()
df_fe['px_area']   = df_fe['px_height'] * df_fe['px_width']          # resolusi total
df_fe['sc_area']   = df_fe['sc_h'] * df_fe['sc_w']                    # luas layar (cm²)
df_fe['batt_per_g'] = df_fe['battery_power'] / df_fe['mobile_wt']     # mAh per gram
df_fe['n_network']  = df_fe[['three_g', 'four_g', 'wifi', 'blue']].sum(axis=1)
df_fe[['px_area', 'sc_area', 'batt_per_g', 'n_network', TARGET]].corr()[TARGET].round(3)""",
                 note="Fitur baru harus bisa dijustifikasi (makna fisik) dan dibuktikan lewat korelasi atau skor CV. Operasi per baris seperti ini aman dilakukan sebelum split."),
            dict(t="X, y & split stratified", code=r"""
X = df_clean.drop(columns=TARGET)
y = df_clean[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(X_train.shape, X_test.shape)                       # (1600, 20) (400, 20)
print(y_train.value_counts(normalize=True).sort_index())  # proporsi sama dengan data asli
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)""",
                 note="<code>stratify=y</code> menjaga proporsi kelas di train dan test. <code>random_state</code> membuat hasil bisa diulang."),
            dict(t="Pipeline preprocessing (fungsi, bukan objek bersama)", code=r"""
def prep(scale=True):
    # buat objek BARU setiap dipanggil supaya tiap model punya imputer/scaler sendiri
    steps = [('imp', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('sc', StandardScaler()))
    return Pipeline(steps)

prep().fit_transform(X_train)[:2].round(2)   # cek: tanpa NaN, mean≈0, std≈1""",
                 note="Jangan memakai satu objek <code>prep</code> yang sama di beberapa Pipeline: fit model kedua akan menimpa scaler model pertama."),
            dict(t="Template data campuran (numerik + kategorikal + missing)", wide=True, code=r"""
df_mix = pd.DataFrame({'umur': [25, np.nan, 40, 35, 52, 28], 'gaji': [5, 7, np.nan, 12, 20, 6],
                       'kota': ['Bdg', 'Jkt', 'Jkt', np.nan, 'Sby', 'Bdg'],
                       'pendidikan': ['S1', 'SMA', 'S2', 'S1', 'S2', 'SMA'], 'beli': [0, 0, 1, 1, 1, 0]})
Xm = df_mix.drop(columns='beli')
num_cols = Xm.select_dtypes('number').columns.tolist()
cat_cols = Xm.select_dtypes(exclude='number').columns.tolist()
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])
model_mix = Pipeline([('prep', preprocessor), ('model', LogisticRegression(max_iter=1000))])
model_mix.fit(Xm, df_mix['beli'])
preprocessor.get_feature_names_out()""",
                 note="Ordinal (S1 &lt; S2) boleh pakai <code>OrdinalEncoder(categories=[['SMA','S1','S2']])</code>. Untuk RF, encoding tetap wajib karena scikit-learn hanya menerima angka."),
            dict(t="Seleksi fitur", code=r"""
mi = mutual_info_classif(SimpleImputer(strategy='median').fit_transform(X_train), y_train,
                         random_state=RANDOM_STATE)
mi = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
mi.head(8).plot.barh(); plt.gca().invert_yaxis(); plt.title('Mutual information (train)'); plt.show()
top_fitur = mi.head(4).index.tolist()      # ['ram', 'battery_power', 'px_width', 'px_height']
# di dalam pipeline (k ikut di-tuning lewat CV):
sel_pipe = Pipeline([('prep', prep()), ('sel', SelectKBest(f_classif, k=4)),
                     ('model', KNeighborsClassifier())])""",
                 note="Hitung skor seleksi dari <b>train</b> saja. Seleksi paling berguna untuk KNN, yang semua fiturnya ikut menghitung jarak."),
            dict(t="Kelas tidak seimbang", code=r"""
LogisticRegression(class_weight='balanced', max_iter=1000)
RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
# metrik: f1_score(..., average='macro'), recall per kelas di classification_report
# oversampling (pip install imbalanced-learn), hanya di train lewat imblearn.pipeline:
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import SMOTE
# ImbPipeline([('prep', prep()), ('smote', SMOTE(random_state=42)), ('model', LogisticRegression())])"""),
            dict(t="Kebutuhan preprocessing per model", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Langkah</th><th>LogReg</th><th>KNN</th><th>Random Forest</th><th>ANN (MLP)</th></tr></thead><tbody>
<tr><td>Imputasi missing</td><td>wajib</td><td>wajib</td><td>wajib*</td><td>wajib</td></tr>
<tr><td>Encoding kategori</td><td>one-hot</td><td>one-hot</td><td>one-hot / ordinal</td><td>one-hot</td></tr>
<tr><td>Scaling</td><td><b>ya</b></td><td><b>wajib</b></td><td>tidak perlu</td><td><b>wajib</b></td></tr>
<tr><td>Sensitif outlier</td><td>sedang</td><td>tinggi</td><td>rendah</td><td>sedang</td></tr>
<tr><td>Fitur tak relevan</td><td>diredam regularisasi</td><td><b>sangat merusak</b></td><td>diabaikan otomatis</td><td>diredam <code>alpha</code></td></tr>
<tr><td>Multikolinearitas</td><td>koefisien tidak stabil</td><td>bobot ganda di jarak</td><td>importance terbagi</td><td>tidak masalah</td></tr>
</tbody></table></div>
<p class="note">* <code>HistGradientBoostingClassifier</code> dan RF di scikit-learn ≥ 1.4 bisa menerima NaN langsung, tetapi imputasi eksplisit lebih mudah dijelaskan.</p>"""),
        ]),
        # ------------------------------------------------------------------ skeleton functions
        dict(id="k-skeleton", title="Skeleton fungsi preprocessing", desc="Kerangka fungsi yang bisa disalin ke notebook mana pun: ganti bagian KONFIGURASI dan isi TODO, sisanya tetap.", cards=[
            dict(t="Pembagian tugas tiap fungsi", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Fungsi</th><th>Tugas</th><th>Kapan dijalankan</th><th>Boleh pakai statistik data?</th></tr></thead><tbody>
<tr><td><code>load_data</code></td><td>baca CSV/URL</td><td>sebelum split</td><td>–</td></tr>
<tr><td><code>inspect_data</code></td><td>tabel kualitas: tipe, missing, nol, unik, duplikat, distribusi kelas</td><td>sebelum split</td><td>hanya untuk dilihat</td></tr>
<tr><td><code>clean_data</code></td><td>perbaikan per baris: duplikat, buang kolom, nilai mustahil → NaN, rapikan teks, tipe</td><td>sebelum split</td><td><b>tidak</b> (tanpa mean/median)</td></tr>
<tr><td><code>add_features</code></td><td>fitur baru dari kolom di baris yang sama</td><td>sebelum split</td><td><b>tidak</b></td></tr>
<tr><td><code>split_data</code></td><td>X, y, train/test stratified</td><td>–</td><td>–</td></tr>
<tr><td><code>get_column_types</code></td><td>daftar kolom numerik &amp; kategorikal</td><td>setelah split (dari train)</td><td>–</td></tr>
<tr><td><code>build_preprocessor</code></td><td>imputasi, scaling, one-hot di <code>ColumnTransformer</code></td><td>di dalam Pipeline</td><td><b>ya</b>, di-fit di train saja</td></tr>
<tr><td><code>build_pipeline</code></td><td>preprocessor (+ seleksi fitur) + model</td><td>di dalam Pipeline</td><td><b>ya</b>, di-fit di train saja</td></tr>
</tbody></table></div>
<p class="note">Aturannya: apa pun yang menghitung angka dari data (median, mean/std, kategori, skor fitur) masuk ke <code>build_preprocessor</code>/<code>build_pipeline</code> agar tidak terjadi leakage.</p>"""),
            dict(t="Skeleton kosong (untuk diisi sendiri)", wide=True, test=False, code=r"""
# ===================== KONFIGURASI =====================
TARGET = '...'            # nama kolom label
RANDOM_STATE = 42
TEST_SIZE = 0.2
DROP_COLS = []            # id, nama, kolom yang membocorkan target
ZERO_AS_NAN = []          # kolom yang nilai 0-nya mustahil


def load_data(path):
    '''Baca data, cetak shape, kembalikan DataFrame.'''
    # TODO: pd.read_csv / pd.read_excel
    pass


def inspect_data(df, target=TARGET):
    '''Kembalikan tabel kualitas per kolom + cetak duplikat & distribusi kelas.'''
    # TODO: dtype, n_missing, pct_missing, n_zero, n_unique
    pass


def clean_data(df, drop_cols=DROP_COLS, zero_as_nan=ZERO_AS_NAN):
    '''Perbaikan per baris saja (tanpa statistik data). Selalu kerja di df.copy().'''
    # TODO: drop_duplicates, drop kolom, 0 → NaN, rapikan teks, perbaiki tipe
    pass


def add_features(df):
    '''Fitur baru dari kolom pada baris yang sama.'''
    # TODO
    pass


def split_data(df, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    '''Pisah X, y lalu train/test dengan stratify=y.'''
    # TODO: return X_train, X_test, y_train, y_test
    pass


def get_column_types(X):
    '''Kembalikan (num_cols, cat_cols).'''
    # TODO: select_dtypes
    pass


def build_preprocessor(num_cols, cat_cols, scale=True):
    '''ColumnTransformer: numerik (imputer + scaler), kategorikal (imputer + one-hot).'''
    # TODO
    pass


def build_pipeline(model, num_cols, cat_cols, scale=True, k_best=None):
    '''Pipeline: preprocessor → (SelectKBest) → model.'''
    # TODO
    pass""", note="Versi terisi ada di empat kartu berikut. Urutan pemanggilan: load → inspect → clean → add_features → split → get_column_types → build_pipeline."),
            dict(t="Konfigurasi + load & inspect", wide=True, code=r"""
# ===================== KONFIGURASI (ubah sesuai dataset) =====================
DATA_PATH = "https://raw.githubusercontent.com/amankharwal/Website-data/master/mobile_prices.csv"
TARGET = 'price_range'
RANDOM_STATE = 42
TEST_SIZE = 0.2
DROP_COLS = []                       # mis. ['id']
ZERO_AS_NAN = ['sc_w', 'px_height']  # 0 tidak mungkin untuk lebar layar / tinggi piksel


def load_data(path):
    df = pd.read_csv(path)
    print('shape:', df.shape)
    return df


def inspect_data(df, target=TARGET):
    info = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'n_missing': df.isna().sum(),
        'pct_missing': (df.isna().mean() * 100).round(1),
        'n_zero': (df == 0).sum(),
        'n_unique': df.nunique(),
    })
    print('duplikat :', df.duplicated().sum())
    if target in df:
        vc = df[target].value_counts()
        print('kelas    :', vc.sort_index().to_dict(), '| rasio max/min =', round(vc.max() / vc.min(), 2))
    return info


df_raw = load_data(DATA_PATH)
inspect_data(df_raw).query('n_missing > 0 or n_zero > 0')""",
                 note="<code>n_zero</code> berguna untuk menemukan 0 yang sebenarnya missing. Untuk fitur biner (0/1), 0 itu wajar, jadi abaikan."),
            dict(t="clean_data & add_features", wide=True, code=r"""
def clean_data(df, drop_cols=DROP_COLS, zero_as_nan=ZERO_AS_NAN):
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.drop(columns=drop_cols, errors='ignore')
    for col in zero_as_nan:
        df[col] = df[col].replace(0, np.nan)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip().str.lower().replace({'': np.nan, '-': np.nan, '?': np.nan})
    # TODO: perbaiki tipe, mis. df['harga'] = pd.to_numeric(df['harga'], errors='coerce')
    print(f'clean_data: {n0} → {len(df)} baris, NaN = {int(df.isna().sum().sum())}')
    return df


def add_features(df):
    df = df.copy()
    # TODO: aktifkan fitur yang bisa dijustifikasi (cek korelasi / skor CV)
    # df['px_area'] = df['px_height'] * df['px_width']
    # df['sc_area'] = df['sc_h'] * df['sc_w']
    # df['batt_per_g'] = df['battery_power'] / df['mobile_wt']
    return df""",
                 note="Fungsi ini sengaja tidak memakai median/mean. Pengisian NaN dilakukan <code>SimpleImputer</code> di Pipeline, setelah split."),
            dict(t="split_data & get_column_types", wide=True, code=r"""
def split_data(df, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f'train {X_train.shape} | test {X_test.shape}')
    return X_train, X_test, y_train, y_test


def get_column_types(X):
    num_cols = X.select_dtypes('number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    # angka yang sebenarnya kode kategori (mis. kode kota 1/2/3): pindahkan manual ke cat_cols.
    # fitur biner 0/1 cukup tetap numerik.
    print(f'{len(num_cols)} numerik, {len(cat_cols)} kategorikal')
    return num_cols, cat_cols"""),
            dict(t="build_preprocessor & build_pipeline", wide=True, code=r"""
def build_preprocessor(num_cols, cat_cols, scale=True, scaler='standard'):
    num_steps = [('imp', SimpleImputer(strategy='median'))]
    if scale:
        num_steps.append(('sc', StandardScaler() if scaler == 'standard' else MinMaxScaler()))
    cat_steps = [('imp', SimpleImputer(strategy='most_frequent')),
                 ('ohe', OneHotEncoder(handle_unknown='ignore'))]
    transformers = [('num', Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append(('cat', Pipeline(cat_steps), cat_cols))
    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)


def build_pipeline(model, num_cols, cat_cols, scale=True, k_best=None):
    steps = [('prep', build_preprocessor(num_cols, cat_cols, scale=scale))]
    if k_best is not None:
        steps.append(('sel', SelectKBest(f_classif, k=k_best)))
    steps.append(('model', model))
    return Pipeline(steps)""",
                 note="Setiap pemanggilan membuat objek baru, jadi aman dipakai untuk banyak model sekaligus. Saran nilai <code>scale</code>: <code>True</code> untuk LogReg/KNN/ANN, <code>False</code> untuk RF/boosting."),
            dict(t="Cara pakai: dari file mentah sampai perbandingan model", wide=True, code=r"""
df_raw = load_data(DATA_PATH)
df_ready = add_features(clean_data(df_raw))
X_train, X_test, y_train, y_test = split_data(df_ready)
num_cols, cat_cols = get_column_types(X_train)

kandidat = {
    'LogReg': build_pipeline(LogisticRegression(max_iter=1000), num_cols, cat_cols),
    'KNN':    build_pipeline(KNeighborsClassifier(n_neighbors=15), num_cols, cat_cols, k_best=4),
    'RF':     build_pipeline(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
                             num_cols, cat_cols, scale=False),
    'ANN':    build_pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True,
                                           random_state=RANDOM_STATE), num_cols, cat_cols),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ringkas = {n: cross_val_score(p, X_train, y_train, cv=skf).mean() for n, p in kandidat.items()}
pd.Series(ringkas, name='acc_cv').sort_values(ascending=False).round(3)""",
                 note="Hasil CV di data ini (parameter default, belum di-tuning): LogReg 0.948, ANN 0.894, KNN 0.883, RF 0.876. Lanjutkan dengan <code>GridSearchCV(kandidat['LogReg'], {'model__C': [...]})</code>, lalu <code>fit(X_train, y_train)</code> dan evaluasi sekali di test (bagian Evaluasi)."),
        ]),
        # ------------------------------------------------------------------ logistic regression
        dict(id="k-lr", title="Regresi Logistik", desc="Model linier untuk peluang kelas. Baseline yang wajib ada.", cards=[
            dict(t="Konsep", html=r"""
<p><b>Biner:</b> z = b₀ + b₁x₁ + … + bₙxₙ, lalu p(y=1) = 1 / (1 + e<sup>−z</sup>) (sigmoid). Prediksi 1 bila p ≥ 0.5.</p>
<p><b>Multikelas</b> (4 kelas harga): satu vektor koefisien per kelas, peluang dihitung dengan <b>softmax</b> sehingga totalnya 1.</p>
<p><b>Regularisasi:</b> <code>C</code> = 1/λ. C kecil → regularisasi kuat (koefisien kecil, cenderung underfit). C besar → lemah (cenderung overfit). <code>l1_ratio=1</code> (L1) bisa membuat koefisien tepat 0 = seleksi fitur.</p>
<p><b>Koefisien:</b> naik 1 satuan fitur (1 std bila di-scale) → log-odds kelas itu naik sebesar b; odds dikali e<sup>b</sup>.</p>"""),
            dict(t="Model dasar", code=r"""
logreg = Pipeline([('prep', prep()), ('model', LogisticRegression(max_iter=1000))])
logreg.fit(X_train, y_train)
print('train :', round(logreg.score(X_train, y_train), 3))
print('test  :', round(logreg.score(X_test, y_test), 3))
proba = logreg.predict_proba(X_test)          # shape (400, 4), tiap baris berjumlah 1
pd.DataFrame(proba[:5], columns=logreg.classes_).round(3)"""),
            dict(t="Tuning C (GridSearchCV)", code=r"""
grid_lr = GridSearchCV(
    Pipeline([('prep', prep()), ('model', LogisticRegression(solver='saga', max_iter=5000))]),
    param_grid={'model__C': [0.01, 0.1, 1, 10, 100, 1000],
                'model__l1_ratio': [0, 1]},           # 0 = L2, 1 = L1 (L1 butuh solver saga/liblinear)
    cv=skf, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)
print(grid_lr.best_params_, round(grid_lr.best_score_, 3))
logreg_best = grid_lr.best_estimator_""",
                 note="scikit-learn &lt; 1.8 (cek <code>sklearn.__version__</code> di Colab): pakai <code>'model__penalty': ['l1', 'l2']</code> dengan <code>solver='saga'</code>, karena <code>l1_ratio</code> hanya berlaku untuk <code>penalty='elasticnet'</code>."),
            dict(t="Interpretasi koefisien", code=r"""
lr = logreg.named_steps['model']
coef = pd.DataFrame(lr.coef_.T, index=X_train.columns,
                    columns=[f'kelas_{k}' for k in lr.classes_])   # (20 fitur × 4 kelas)
coef.sort_values('kelas_3', ascending=False).round(2).head(6)
np.exp(coef['kelas_3']).sort_values(ascending=False).head(4)       # odds ratio per +1 std
coef['kelas_3'].sort_values().plot.barh(figsize=(6, 6))
plt.title('Koefisien LogReg untuk kelas 3 (very high)'); plt.axvline(0, c='k', lw=.8); plt.show()""",
                 note="Koefisien hanya bisa dibandingkan antar fitur bila fitur sudah di-scale. Tanda + berarti fitur itu menaikkan peluang kelas tersebut."),
        ]),
        # ------------------------------------------------------------------ KNN
        dict(id="k-knn", title="K-Nearest Neighbors", desc="Klasifikasi berdasarkan k tetangga terdekat. Hasilnya sangat bergantung pada skala dan pilihan fitur.", cards=[
            dict(t="Konsep", html=r"""
<p>Untuk titik baru: hitung jarak ke semua data latih, ambil <b>k</b> terdekat, kelas terbanyak = prediksi (<code>weights='distance'</code>: tetangga dekat bersuara lebih besar).</p>
<p><b>Jarak:</b> Euclidean (<code>p=2</code>) √Σ(xᵢ−yᵢ)², Manhattan (<code>p=1</code>) Σ|xᵢ−yᵢ|.</p>
<p><b>Lazy learner:</b> <code>fit</code> hanya menyimpan data; perhitungan terjadi saat <code>predict</code>.</p>
<p><b>k kecil</b> → batas keputusan bergerigi, overfit (variance tinggi). <b>k besar</b> → terlalu halus, underfit (bias tinggi). Pilih k ganjil agar voting tidak seri (untuk 2 kelas).</p>
<p><b>Curse of dimensionality:</b> makin banyak fitur, jarak antar titik makin mirip, sehingga fitur tak relevan menenggelamkan fitur penting.</p>"""),
            dict(t="Coba beberapa k, dengan & tanpa scaling", code=r"""
ks = list(range(1, 32, 2))
skor = {'tanpa scaling': [], 'dengan scaling': []}
for k in ks:
    for nama, sc in [('tanpa scaling', False), ('dengan scaling', True)]:
        m = Pipeline([('prep', prep(scale=sc)), ('model', KNeighborsClassifier(n_neighbors=k))])
        skor[nama].append(cross_val_score(m, X_train, y_train, cv=skf).mean())
hasil_k = pd.DataFrame(skor, index=ks)
hasil_k.plot(marker='o'); plt.xlabel('k'); plt.ylabel('accuracy (5-fold CV)')
plt.title('KNN: pengaruh k dan scaling'); plt.show()
hasil_k.idxmax(), hasil_k.max().round(3)""",
                 note="Pada data ini KNN <b>tanpa</b> scaling justru jauh lebih baik (CV terbaik 0.922 vs 0.590). Tanpa scaling, jarak didominasi <code>ram</code> (skala ribuan) yang kebetulan fitur terkuat. Setelah scaling, 16 fitur yang hampir tak berhubungan dengan harga ikut berbobot sama dan menenggelamkan <code>ram</code>. Solusinya scaling + seleksi fitur (kartu berikut)."),
            dict(t="KNN + seleksi fitur + GridSearchCV", code=r"""
knn_pipe = Pipeline([('prep', prep()), ('sel', SelectKBest(f_classif)),
                     ('model', KNeighborsClassifier())])
grid_knn = GridSearchCV(knn_pipe, {
    'sel__k': [2, 3, 4, 6, 8, 'all'],
    'model__n_neighbors': [5, 9, 15, 21, 31],
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2]}, cv=skf, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)
print(grid_knn.best_params_, round(grid_knn.best_score_, 3))
knn_best = grid_knn.best_estimator_
X_train.columns[knn_best.named_steps['sel'].get_support()].tolist()   # fitur terpilih"""),
            dict(t="Decision boundary 2 fitur", code=r"""
dua = ['ram', 'battery_power']
knn2 = Pipeline([('sc', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=15))])
knn2.fit(X_train[dua], y_train)
disp = DecisionBoundaryDisplay.from_estimator(knn2, X_train[dua], response_method='predict', alpha=.35)
sns.scatterplot(data=X_train.assign(kelas=y_train), x=dua[0], y=dua[1], hue='kelas',
                palette='tab10', s=12, ax=disp.ax_)
plt.title('Decision boundary KNN (k=15)'); plt.show()""",
                 note="Ganti <code>n_neighbors</code> menjadi 1 untuk melihat batas yang bergerigi (overfit)."),
        ]),
        # ------------------------------------------------------------------ RF
        dict(id="k-rf", title="Random Forest", desc="Ensemble bagging dari decision tree. Kuat tanpa banyak preprocessing.", cards=[
            dict(t="Konsep", html=r"""
<p><b>Decision tree:</b> membagi data dengan pertanyaan ya/tidak (mis. <code>ram &lt;= 2200</code>) yang paling menurunkan impurity (<b>Gini</b> = 1 − Σpᵢ², atau entropy).</p>
<p><b>Random Forest</b> = banyak tree, masing-masing dilatih pada <b>bootstrap sample</b> (sampling dengan pengembalian) dan hanya melihat <b>subset fitur acak</b> di tiap split (<code>max_features='sqrt'</code>). Prediksi = voting mayoritas.</p>
<p>Tree tunggal punya variance tinggi; merata-ratakan banyak tree yang <b>tidak saling berkorelasi</b> menurunkan variance tanpa menaikkan bias.</p>
<p><b>OOB score:</b> tiap tree tidak melihat ±37% data; data itu dipakai sebagai validasi gratis.</p>"""),
            dict(t="Model dasar + OOB", code=r"""
rf = Pipeline([('prep', prep(scale=False)),
               ('model', RandomForestClassifier(n_estimators=300, oob_score=True,
                                                random_state=RANDOM_STATE, n_jobs=-1))])
rf.fit(X_train, y_train)
print('train :', round(rf.score(X_train, y_train), 3))     # biasanya 1.0
print('OOB   :', round(rf.named_steps['model'].oob_score_, 3))
print('test  :', round(rf.score(X_test, y_test), 3))""",
                 note="Akurasi train 1.0 wajar untuk RF. Yang penting jarak antara OOB/CV dan test kecil."),
            dict(t="Feature importance (MDI vs permutation)", code=r"""
imp = pd.Series(rf.named_steps['model'].feature_importances_, index=X_train.columns)
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm = pd.Series(perm.importances_mean, index=X_train.columns)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
imp.sort_values().tail(10).plot.barh(ax=ax[0], title='MDI (impurity, dari train)')
perm.sort_values().tail(10).plot.barh(ax=ax[1], title='Permutation (penurunan acc di test)')
plt.tight_layout(); plt.show()""",
                 note="MDI cenderung melebihkan fitur kontinu dengan banyak nilai unik. Permutation importance lebih jujur: seberapa turun accuracy bila kolom itu diacak."),
            dict(t="Tuning (RandomizedSearchCV)", code=r"""
rs_rf = RandomizedSearchCV(
    Pipeline([('prep', prep(scale=False)), ('model', RandomForestClassifier(random_state=RANDOM_STATE))]),
    param_distributions={'model__n_estimators': [200, 300, 500],
                         'model__max_depth': [None, 8, 12, 20],
                         'model__min_samples_leaf': [1, 2, 4],
                         'model__max_features': ['sqrt', 0.5, None]},
    n_iter=12, cv=skf, scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1)
rs_rf.fit(X_train, y_train)
print(rs_rf.best_params_, round(rs_rf.best_score_, 3))
rf_best = rs_rf.best_estimator_""", test="slow",
                 note="Di data ini <code>max_features=0.5</code> menang (CV 0.89 vs 0.876 default) karena hanya sedikit fitur yang informatif: dengan <code>'sqrt'</code> (4 dari 20 fitur per split), banyak split tidak sempat melihat <code>ram</code>."),
        ]),
        # ------------------------------------------------------------------ ANN
        dict(id="k-ann", title="ANN (Multilayer Perceptron)", desc="Jaringan saraf tiruan: MLPClassifier untuk tugas, Keras bila diminta deep learning.", cards=[
            dict(t="Konsep", html=r"""
<p><b>Neuron:</b> a = f(w·x + b). <b>Layer:</b> input (jumlah fitur) → hidden (mis. 64 → 32) → output (jumlah kelas, aktivasi <b>softmax</b>).</p>
<p><b>Aktivasi hidden:</b> <code>relu</code> max(0, z) (default), <code>tanh</code>, <code>logistic</code>.</p>
<p><b>Training:</b> forward pass → loss (cross-entropy) → <b>backpropagation</b> menghitung gradien → optimizer (<code>adam</code>, <code>sgd</code>) memperbarui bobot. Satu <b>epoch</b> = seluruh data train sekali lewat.</p>
<p><b>Mencegah overfit:</b> <code>alpha</code> (regularisasi L2), <code>early_stopping=True</code> (berhenti saat skor validasi tidak naik), jaringan lebih kecil.</p>
<p><b>Wajib scaling:</b> gradien tidak stabil bila skala fitur berbeda jauh.</p>"""),
            dict(t="MLPClassifier", code=r"""
ann = Pipeline([('prep', prep()),
                ('model', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                        alpha=1e-3, learning_rate_init=1e-3, max_iter=1000,
                                        early_stopping=True, validation_fraction=0.1,
                                        random_state=RANDOM_STATE))])
ann.fit(X_train, y_train)
print('train :', round(ann.score(X_train, y_train), 3))
print('test  :', round(ann.score(X_test, y_test), 3))
print('epoch :', ann.named_steps['model'].n_iter_)"""),
            dict(t="Kurva loss & skor validasi", code=r"""
mlp = ann.named_steps['model']
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(mlp.loss_curve_); ax[0].set(title='Loss train', xlabel='epoch')
ax[1].plot(mlp.validation_scores_); ax[1].set(title='Accuracy validasi', xlabel='epoch')
plt.tight_layout(); plt.show()""",
                 note="Loss train terus turun tetapi skor validasi datar/turun → overfit. Keduanya masih naik di epoch terakhir → tambah <code>max_iter</code>."),
            dict(t="Tuning MLP", code=r"""
grid_ann = GridSearchCV(
    Pipeline([('prep', prep()), ('model', MLPClassifier(max_iter=2000, early_stopping=True,
                                                        random_state=RANDOM_STATE))]),
    {'model__hidden_layer_sizes': [(32,), (64,), (64, 32)],
     'model__alpha': [1e-4, 1e-2, 1],
     'model__learning_rate_init': [1e-3, 1e-2]},
    cv=skf, scoring='accuracy', n_jobs=-1)
grid_ann.fit(X_train, y_train)
print(grid_ann.best_params_, round(grid_ann.best_score_, 3))
ann_best = grid_ann.best_estimator_""", test="slow"),
            dict(t="Versi Keras (opsional)", wide=True, code=r"""
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(RANDOM_STATE)
pre = prep().fit(X_train)                          # fit scaler di train saja
Xtr_s, Xte_s = pre.transform(X_train), pre.transform(X_test)
net = keras.Sequential([
    keras.Input(shape=(Xtr_s.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax'),   # 4 kelas
])
net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = net.fit(Xtr_s, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=0,
               callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
pd.DataFrame(hist.history)[['accuracy', 'val_accuracy']].plot(); plt.xlabel('epoch'); plt.show()
pred_keras = net.predict(Xte_s, verbose=0).argmax(axis=1)
accuracy_score(y_test, pred_keras)""", test="slow",
                 note="Label integer 0..3 → <code>sparse_categorical_crossentropy</code>. Label one-hot → <code>categorical_crossentropy</code>. Klasifikasi biner → 1 neuron <code>sigmoid</code> + <code>binary_crossentropy</code>."),
        ]),
        # ------------------------------------------------------------------ ensemble
        dict(id="k-ens", title="Ensemble", desc="Menggabungkan beberapa model agar kesalahannya saling menutupi.", cards=[
            dict(t="Jenis ensemble", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Jenis</th><th>Ide</th><th>Menurunkan</th><th>Kelas scikit-learn</th></tr></thead><tbody>
<tr><td><b>Voting hard</b></td><td>tiap model memilih label, mayoritas menang</td><td>variance</td><td><code>VotingClassifier(voting='hard')</code></td></tr>
<tr><td><b>Voting soft</b></td><td>rata-rata <code>predict_proba</code>, kelas dengan peluang tertinggi menang (biasanya lebih baik dari hard)</td><td>variance</td><td><code>VotingClassifier(voting='soft')</code></td></tr>
<tr><td><b>Stacking</b></td><td>prediksi model dasar (dari CV) jadi fitur untuk <i>meta-model</i></td><td>bias &amp; variance</td><td><code>StackingClassifier</code></td></tr>
<tr><td><b>Bagging</b></td><td>model yang sama dilatih di banyak bootstrap sample, lalu voting (RF = bagging tree + fitur acak)</td><td>variance</td><td><code>BaggingClassifier</code>, <code>RandomForestClassifier</code></td></tr>
<tr><td><b>Boosting</b></td><td>model lemah dilatih berurutan, tiap model fokus pada kesalahan sebelumnya</td><td>bias</td><td><code>AdaBoostClassifier</code>, <code>GradientBoostingClassifier</code>, <code>HistGradientBoostingClassifier</code></td></tr>
</tbody></table></div>
<p class="note">Ensemble menguntungkan bila model dasarnya <b>cukup baik dan beragam</b> (kesalahannya tidak sama). Menggabungkan model yang sangat lemah tidak menolong.</p>"""),
            dict(t="VotingClassifier (hard vs soft)", code=r"""
base = [('lr', Pipeline([('prep', prep()), ('m', LogisticRegression(max_iter=1000))])),
        ('knn', Pipeline([('prep', prep()), ('sel', SelectKBest(f_classif, k=4)),
                          ('m', KNeighborsClassifier(n_neighbors=15))])),
        ('rf', Pipeline([('prep', prep(scale=False)),
                         ('m', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]))]
vote_hard = VotingClassifier(base, voting='hard').fit(X_train, y_train)
vote_soft = VotingClassifier(base, voting='soft').fit(X_train, y_train)   # bisa diberi weights=[2, 1, 1]
print('hard :', round(vote_hard.score(X_test, y_test), 3))
print('soft :', round(vote_soft.score(X_test, y_test), 3))""",
                 note="<code>voting='soft'</code> butuh semua model punya <code>predict_proba</code>. Model di dalam ensemble di-<i>clone</i>, jadi objek <code>base</code> boleh dipakai ulang."),
            dict(t="StackingClassifier", code=r"""
stack = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1)                      # meta-model dilatih dari prediksi out-of-fold
stack.fit(X_train, y_train)
print('stacking :', round(stack.score(X_test, y_test), 3))""", test="slow"),
            dict(t="Bagging & Boosting", code=r"""
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=300,     # bagging tree tanpa fitur acak
                        random_state=RANDOM_STATE, n_jobs=-1)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200,
                         learning_rate=0.5, random_state=RANDOM_STATE)
gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                 random_state=RANDOM_STATE)
hgb = HistGradientBoostingClassifier(learning_rate=0.1, random_state=RANDOM_STATE)  # cepat, terima NaN
for nama, m in [('Bagging tree', bag), ('AdaBoost', ada), ('GradBoost', gb), ('HistGB', hgb)]:
    m.fit(X_train.fillna(X_train.median()), y_train)
    print(f'{nama:12s}', round(m.score(X_test.fillna(X_train.median()), y_test), 3))""",
                 note="AdaBoost dengan stump bawaan (<code>max_depth=1</code>) hanya 0.49 di data 4 kelas ini; dengan tree <code>max_depth=3</code> naik ke 0.943. GradientBoosting 0.927, HistGB 0.925, Bagging tree 0.89. Learning rate kecil + estimator banyak = lebih stabil tapi lambat."),
        ]),
        # ------------------------------------------------------------------ evaluation
        dict(id="k-eval", title="Evaluasi di test set", desc="Bandingkan semua model dengan metrik dan data uji yang sama.", cards=[
            dict(t="Evaluasi satu model", code=r"""
def evaluasi(nama, model, tampil=True):
    pred = model.predict(X_test)
    if tampil:
        print(f'=== {nama} ===  accuracy = {accuracy_score(y_test, pred):.3f}')
        print(classification_report(y_test, pred, digits=3,
                                    target_names=['low', 'medium', 'high', 'very high']))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='Blues', colorbar=False)
        plt.title(nama); plt.grid(False); plt.show()
    return {'model': nama,
            'acc_train': model.score(X_train, y_train),
            'acc_test': accuracy_score(y_test, pred),
            'precision_macro': precision_score(y_test, pred, average='macro'),
            'recall_macro': recall_score(y_test, pred, average='macro'),
            'f1_macro': f1_score(y_test, pred, average='macro')}

evaluasi('Logistic Regression', logreg)"""),
            dict(t="Tabel perbandingan semua model", wide=True, code=r"""
models = {'Logistic Regression': logreg_best, 'KNN (scaled + SelectKBest)': knn_best,
          'Random Forest': rf, 'ANN (MLP)': ann,
          'Voting soft': vote_soft, 'Voting hard': vote_hard}
hasil = pd.DataFrame([evaluasi(n, m, tampil=False) for n, m in models.items()]).set_index('model')
hasil['gap'] = hasil['acc_train'] - hasil['acc_test']          # besar → overfit
hasil = hasil.sort_values('acc_test', ascending=False).round(3)
display(hasil)
hasil['acc_test'].plot.barh(xlim=(0.5, 1)); plt.gca().invert_yaxis()
plt.title('Accuracy test per model'); plt.xlabel('accuracy'); plt.show()""",
                 note="Model yang sudah di-fit (atau <code>best_estimator_</code> dari GridSearchCV) tidak perlu di-fit ulang. <code>display</code> tersedia di Jupyter/Colab; di skrip biasa pakai <code>print</code>."),
            dict(t="Confusion matrix berdampingan", wide=True, code=r"""
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
for ax, (nama, m) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(f'{nama}\nacc={m.score(X_test, y_test):.3f}', fontsize=9); ax.grid(False)
plt.tight_layout(); plt.show()"""),
            dict(t="ROC-AUC multikelas (one-vs-rest)", code=r"""
proba = logreg_best.predict_proba(X_test)
print('AUC macro OvR:', round(roc_auc_score(y_test, proba, multi_class='ovr', average='macro'), 4))
fig, ax = plt.subplots(figsize=(5, 5))
for i, kelas in enumerate(logreg_best.classes_):
    RocCurveDisplay.from_predictions(y_test == kelas, proba[:, i], name=f'kelas {kelas}', ax=ax)
ax.plot([0, 1], [0, 1], 'k--'); ax.set_title('ROC per kelas (LogReg)'); plt.show()""",
                 note="AUC dihitung dari <b>peluang</b>, bukan label. Model tanpa <code>predict_proba</code> (Voting hard) tidak bisa dihitung AUC-nya."),
            dict(t="Cross-validation semua model (lebih stabil)", code=r"""
cv_rows = []
for nama, m in models.items():
    s = cross_validate(m, X_train, y_train, cv=skf, scoring=['accuracy', 'f1_macro'], n_jobs=-1)
    cv_rows.append({'model': nama, 'acc_cv_mean': s['test_accuracy'].mean(),
                    'acc_cv_std': s['test_accuracy'].std(), 'f1_cv_mean': s['test_f1_macro'].mean()})
pd.DataFrame(cv_rows).set_index('model').sort_values('acc_cv_mean', ascending=False).round(3)""",
                 note="Skor test dari 400 baris bisa berubah ±1–2% hanya karena split. Rata-rata ± std dari CV menunjukkan apakah selisih antar model benar-benar berarti."),
            dict(t="Membaca metrik multikelas", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Istilah</th><th>Arti</th><th>Kapan dipakai</th></tr></thead><tbody>
<tr><td><b>Precision kelas k</b></td><td>dari yang diprediksi kelas k, berapa persen benar</td><td>salah tebak sebagai k itu mahal</td></tr>
<tr><td><b>Recall kelas k</b></td><td>dari yang sebenarnya kelas k, berapa persen tertangkap</td><td>kelewatan kelas k itu mahal</td></tr>
<tr><td><b>F1</b></td><td>2·P·R / (P+R)</td><td>satu angka yang menyeimbangkan P dan R</td></tr>
<tr><td><b>macro avg</b></td><td>rata-rata biasa antar kelas (semua kelas setara)</td><td>kelas tidak seimbang, peduli kelas minoritas</td></tr>
<tr><td><b>weighted avg</b></td><td>rata-rata dibobot jumlah data per kelas (<i>support</i>)</td><td>ingin mencerminkan distribusi data</td></tr>
<tr><td><b>Confusion matrix</b></td><td>baris = label asli, kolom = prediksi; diagonal = benar</td><td>melihat kelas mana yang tertukar</td></tr>
</tbody></table></div>
<p class="note">Kelas harga bersifat <b>ordinal</b>: kesalahan yang sehat ada di kelas tetangga (1↔2). Kesalahan 0↔3 menandakan model bermasalah.</p>"""),
        ]),
        # ------------------------------------------------------------------ analysis
        dict(id="k-analisis", title="Analisis & menulis kesimpulan", desc="Bagian yang membedakan nilai: menjelaskan mengapa, bukan hanya angka.", cards=[
            dict(t="Contoh hasil pada mobile_prices", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Model</th><th>Hyperparameter</th><th>Acc train</th><th>Acc CV (5-fold)</th><th>Acc test</th><th>F1 macro test</th></tr></thead><tbody>
<tr><td><b>Logistic Regression</b></td><td>scaled, <code>C=1</code>, L1 (<code>l1_ratio=1</code>)</td><td>0.983</td><td>0.966 ± 0.004</td><td><b>0.980</b></td><td><b>0.980</b></td></tr>
<tr><td>Stacking (LR + KNN + RF → LR)</td><td><code>cv=5</code></td><td>–</td><td>–</td><td>0.970</td><td>–</td></tr>
<tr><td>Voting soft (LR + KNN + RF)</td><td>bobot sama</td><td>0.996</td><td>0.938 ± 0.019</td><td>0.962</td><td>0.962</td></tr>
<tr><td>Voting hard (LR + KNN + RF)</td><td>bobot sama</td><td>0.991</td><td>0.927 ± 0.023</td><td>0.952</td><td>0.952</td></tr>
<tr><td>AdaBoost</td><td>tree <code>max_depth=3</code>, 200 estimator</td><td>–</td><td>–</td><td>0.943</td><td>–</td></tr>
<tr><td>ANN (MLP)</td><td>(64, 32), <code>alpha=1e-3</code>, early stopping</td><td>0.983</td><td>0.894 ± 0.020</td><td>0.925</td><td>0.925</td></tr>
<tr><td>KNN</td><td>scaled + SelectKBest(k=4), <code>k=15</code>, <code>p=1</code>, distance</td><td>1.000</td><td>0.904 ± 0.013</td><td>0.908</td><td>0.908</td></tr>
<tr><td>Random Forest</td><td>300 tree, default</td><td>1.000</td><td>0.876 ± 0.030</td><td>0.880</td><td>0.879</td></tr>
<tr><td>KNN tanpa scaling</td><td><code>k=17</code>, semua fitur</td><td>–</td><td>0.922</td><td>–</td><td>–</td></tr>
<tr><td>KNN scaling, semua fitur</td><td><code>k=29</code></td><td>–</td><td>0.590</td><td>–</td><td>–</td></tr>
</tbody></table></div>
<p class="note">Hasil menjalankan semua kartu di tab ini (split 80/20, <code>stratify=y</code>, <code>random_state=42</code>, scikit-learn 1.9). ROC-AUC macro LogReg = 0.999. Kesalahan LogReg hanya 8 dari 400, semuanya ke kelas tetangga (1→2: 3, 2→3: 2, 3→2: 2, 1→0: 1). Angka bisa sedikit berbeda di versi library lain.</p>"""),
            dict(t="Kelas mana yang tertukar?", code=r"""
best_name = hasil.index[0]
best = models[best_name]
pred_best = best.predict(X_test)
cm = pd.DataFrame(confusion_matrix(y_test, pred_best),
                  index=[f'asli {k}' for k in range(4)], columns=[f'pred {k}' for k in range(4)])
salah = cm.stack()[lambda s: s > 0]
salah = salah[[a.split()[1] != b.split()[1] for a, b in salah.index]].sort_values(ascending=False)
print(best_name); print(salah.head())               # pasangan kesalahan terbanyak
# baris yang salah diprediksi: apa yang membuatnya sulit?
X_err = X_test.assign(asli=y_test, pred=pred_best)[pred_best != y_test]
X_err[['ram', 'battery_power', 'px_width', 'asli', 'pred']].head()"""),
            dict(t="Overfitting: train vs CV vs test", code=r"""
cv_df = pd.DataFrame(cv_rows).set_index('model')
ov = hasil[['acc_train', 'acc_test']].join(cv_df['acc_cv_mean'])
ov.plot.bar(figsize=(9, 4), ylim=(0.5, 1.02), rot=20)
plt.title('Train vs CV vs Test'); plt.ylabel('accuracy'); plt.tight_layout(); plt.show()""",
                 note="Train ≫ test → overfit (RF, tree tanpa batas). Train ≈ test tapi keduanya rendah → underfit (k terlalu besar, C terlalu kecil). CV ≈ test → estimasi bisa dipercaya."),
            dict(t="Pola menjawab \"model mana yang lebih baik dan mengapa\"", wide=True, html=r"""
<ol>
<li><b>Angka:</b> sebut model terbaik dan metriknya di test set (accuracy, F1 macro), bandingkan dengan model lain dan dengan CV.</li>
<li><b>Mengapa (sifat data × sifat model):</b> hubungkan dengan temuan EDA, misalnya hubungan fitur–target yang linier/monoton, jumlah fitur tak relevan, skala fitur.</li>
<li><b>Pengaruh preprocessing:</b> scaling, seleksi fitur, imputasi: apa efeknya pada tiap model.</li>
<li><b>Overfitting / stabilitas:</b> selisih train–test, std CV.</li>
<li><b>Pola kesalahan &amp; batasan:</b> kelas mana yang tertukar, dan saran perbaikan.</li>
</ol>
<p class="lead-s">Contoh (angka dari kartu sebelumnya):</p>
<blockquote>Regresi Logistik adalah model terbaik dengan accuracy test 0.980 dan F1 macro 0.980, di atas Stacking (0.970), Voting soft (0.962), ANN (0.925), KNN (0.908), dan Random Forest (0.880); urutan yang sama terlihat pada 5-fold CV (0.966 ± 0.004), jadi keunggulannya bukan kebetulan split. Dari EDA, <code>price_range</code> hampir ditentukan secara monoton oleh <code>ram</code> (korelasi 0.92) ditambah <code>battery_power</code> dan resolusi layar, sehingga batas antar kelas mendekati linier, cocok dengan asumsi regresi logistik, dan regularisasi L1 menekan fitur yang tidak relevan. KNN sangat bergantung pada preprocessing: dengan scaling pada 20 fitur accuracy CV-nya hanya 0.59 karena 16 fitur yang tidak berhubungan dengan harga ikut menentukan jarak, dan naik ke 0.90 setelah dipilih 4 fitur teratas. Random Forest mencapai accuracy train 1.000 tetapi test 0.880 (overfit), karena dengan <code>max_features='sqrt'</code> banyak split tidak melihat <code>ram</code>. Semua 8 kesalahan Regresi Logistik terjadi antar kelas bertetangga (mis. medium→high), sehingga model sudah menangkap urutan harga dengan baik.</blockquote>"""),
            dict(t="Checklist sebelum submit", html=r"""
<ul>
<li>Sel pertama: nama &amp; NIM anggota</li>
<li><code>random_state</code> tetap di split, CV, dan model</li>
<li>Split <b>sebelum</b> imputasi/scaling/seleksi; semua transformasi di dalam Pipeline</li>
<li>Tuning hanya pakai CV di train; test dipakai sekali di akhir</li>
<li>Untuk tiap model: accuracy, confusion matrix, classification report</li>
<li>Tabel perbandingan + paragraf kesimpulan 3–5 kalimat</li>
<li>Restart &amp; Run all, pastikan output tampil, lalu unduh <code>.ipynb</code></li>
</ul>"""),
        ]),
        # ------------------------------------------------------------------ theory & errors
        dict(id="k-teori", title="Teori singkat & error umum", desc="Bahan soal teori dan cara keluar dari error yang paling sering muncul.", cards=[
            dict(t="Jawaban singkat yang sering ditanya", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Pertanyaan</th><th>Jawaban inti</th></tr></thead><tbody>
<tr><td>Mengapa <code>stratify=y</code>?</td><td>Proporsi kelas di train dan test sama dengan data asli, sehingga evaluasi adil (penting bila kelas tidak seimbang).</td></tr>
<tr><td>Mengapa scaler di-fit di train saja?</td><td>Mean/std dari test adalah informasi masa depan; memakainya = <b>data leakage</b> → skor terlalu optimis.</td></tr>
<tr><td>Mengapa KNN/LogReg/ANN butuh scaling, RF tidak?</td><td>KNN memakai jarak, LogReg/ANN memakai gradien &amp; regularisasi yang sensitif skala. Tree hanya membandingkan ambang per fitur, urutan nilai tidak berubah oleh scaling.</td></tr>
<tr><td>Pengaruh k pada KNN?</td><td>k kecil → variance tinggi (overfit); k besar → bias tinggi (underfit). Pilih lewat CV.</td></tr>
<tr><td>Arti <code>C</code> di LogReg?</td><td>Kebalikan kekuatan regularisasi. C kecil = koefisien ditekan (lebih sederhana).</td></tr>
<tr><td>Bagging vs boosting?</td><td>Bagging: paralel, data bootstrap, menurunkan variance (RF). Boosting: berurutan, fokus pada kesalahan, menurunkan bias (AdaBoost, GB).</td></tr>
<tr><td>Hard vs soft voting?</td><td>Hard: mayoritas label. Soft: rata-rata peluang; memanfaatkan keyakinan model, biasanya lebih akurat.</td></tr>
<tr><td>Mengapa RF acc train = 1.0?</td><td>Tree tumbuh penuh hingga daun murni sehingga menghafal train; generalisasi dinilai dari OOB/CV/test.</td></tr>
<tr><td>Mengapa ANN hasilnya berubah tiap run?</td><td>Bobot awal acak dan urutan batch; tetapkan <code>random_state</code>.</td></tr>
<tr><td>Accuracy vs F1 macro?</td><td>Accuracy cukup bila kelas seimbang; F1 macro memberi bobot sama ke tiap kelas sehingga kelas minoritas terlihat.</td></tr>
<tr><td>Mengapa tuning tidak memakai test set?</td><td>Test yang dipakai memilih model ikut "dilatih"; skornya tidak lagi menggambarkan data baru.</td></tr>
</tbody></table></div>"""),
            dict(t="Pesan error & warning", wide=True, html=r"""
<div class="tbl"><table><thead><tr><th>Pesan</th><th>Penyebab</th><th>Perbaikan</th></tr></thead><tbody>
<tr><td><code>ConvergenceWarning: lbfgs failed to converge</code></td><td>LogReg/MLP belum konvergen</td><td>scaling fitur, <code>max_iter=1000</code> atau lebih</td></tr>
<tr><td><code>ValueError: Input X contains NaN</code></td><td>ada missing, model tidak menerima NaN</td><td><code>SimpleImputer</code> di Pipeline</td></tr>
<tr><td><code>ValueError: could not convert string to float</code></td><td>kolom teks masuk ke model</td><td><code>OneHotEncoder</code> lewat <code>ColumnTransformer</code></td></tr>
<tr><td><code>Invalid parameter 'C' for estimator Pipeline</code></td><td>nama parameter grid tanpa prefix langkah</td><td><code>'model__C'</code> (nama langkah + dua garis bawah)</td></tr>
<tr><td><code>AttributeError: predict_proba is not available when voting='hard'</code></td><td>hard voting tidak punya peluang</td><td><code>voting='soft'</code></td></tr>
<tr><td><code>FutureWarning: 'penalty' was deprecated in version 1.8</code></td><td>scikit-learn baru</td><td>pakai <code>l1_ratio</code> (0 = L2, 1 = L1) dan <code>C</code></td></tr>
<tr><td><code>The least populated class in y has only 1 members</code></td><td><code>stratify</code>/StratifiedKFold dengan kelas sangat kecil</td><td>gabungkan kelas langka atau kurangi <code>n_splits</code></td></tr>
<tr><td><code>X has 4 features, but ... is expecting 20 features</code></td><td>kolom saat predict berbeda dengan saat fit</td><td>pakai daftar kolom yang sama (<code>X_test[cols]</code>)</td></tr>
<tr><td>Accuracy 1.0 di test</td><td>target bocor ke fitur, atau test = train</td><td>cek <code>X.columns</code> tidak berisi target / turunannya</td></tr>
</tbody></table></div>"""),
        ]),
    ],
)


def run_tests(quick=False, collect=None):
    """Execute every code card in order in one namespace. Returns number of failures.
    If `collect` is a dict, the final namespace is copied into it (used by build_page for real numbers)."""
    import time
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {"display": print}
    exec(KLASIFIKASI["fixture"], ns)
    bad = 0
    for s in KLASIFIKASI["sections"]:
        for c in s["cards"]:
            if "code" not in c or c.get("test") is False or (quick and c.get("test") == "slow"):
                continue
            t0 = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    exec(c["code"].strip("\n"), ns)
                except Exception as e:
                    bad += 1
                    print(f"[klasifikasi] FAIL {c['t']}: {type(e).__name__}: {e}")
                seen = set()
                for x in w:
                    msg = str(x.message)
                    if "non-interactive" in msg or msg in seen:
                        continue
                    seen.add(msg)
                    print(f"[klasifikasi] WARN {c['t']}: {x.category.__name__}: {msg[:160]}")
            plt.close("all")
            print(f"  ok {time.time() - t0:5.1f}s  {c['t']}")
    if collect is not None:
        collect.update(ns)
    return bad


if __name__ == "__main__":
    import os
    import sys
    ns = {}
    n_bad = run_tests(quick=bool(os.environ.get("QUICK")), collect=ns)
    if "ringkas" in ns:
        print("ringkas", {k: round(v, 3) for k, v in ns["ringkas"].items()})
    if "hasil_k" in ns:
        print(ns["hasil_k"].round(3).to_string())
    if "hasil" in ns:
        print(ns["hasil"].to_string())
    if "cv_rows" in ns:
        import pandas as pd
        print(pd.DataFrame(ns["cv_rows"]).round(3).to_string())
    for g in ["grid_lr", "grid_knn", "rs_rf", "grid_ann"]:
        if g in ns:
            print(g, ns[g].best_params_, round(ns[g].best_score_, 3))
    sys.exit(1 if n_bad else 0)
