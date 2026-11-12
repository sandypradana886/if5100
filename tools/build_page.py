"""Generate index.html for the IF5100 practicum guide from the content defined below.

Usage (from the project root):
    python tools/build_page.py

Edit the section(...) / snip(...) calls to change content. Styling lives in src/style.css and
behaviour in src/main.js. Chart illustrations come from tools/figs.json (re-render them with
tools/make_figs.py). Every snippet is also written to tools/_snippets.py for a quick syntax check."""
import html
import sys
import textwrap

import json

from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
OUT_HTML, OUT_PY, FIGS_JSON = ROOT / "index.html", TOOLS / "_snippets.py", TOOLS / "figs.json"
FIGS = json.load(open(FIGS_JSON, encoding="utf-8"))

sections = []          # (id, title, tag, intro, [blocks], tab)
all_snippets = []
CUR_TAB = ["viz"]


def snip(title, code, note="", lib="", fig=None):
    code = textwrap.dedent(code).strip("\n")
    all_snippets.append((title, code))
    return {"kind": "code", "title": title, "code": code, "note": note, "lib": lib, "fig": fig}


def rich(title, body_html, fig=None):
    return {"kind": "html", "title": title, "body": body_html, "fig": fig}


def section(sid, title, tag, intro, blocks):
    sections.append((sid, title, tag, intro, blocks, CUR_TAB[0]))


# ---------------------------------------------------------------------------
section("setup", "Setup & import data", "", "Sel pertama setiap notebook praktikum. Pilih salah satu cara masuknya data.", [
    snip("Import standar", """
        %matplotlib inline
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        sns.set_theme()                 # tema seaborn (sama dengan sns.set())
    """),
    snip("Opsi A · Mount Google Drive", """
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_DIR = '/content/drive/My Drive/IF5100/visdat/'
        df = pd.read_csv(DATA_DIR + 'data.csv')
    """, "Upload sekali di Drive, aman walau runtime di-reset."),
    snip("Opsi B · Upload langsung", """
        from google.colab import files
        uploaded = files.upload()       # pilih file (boleh beberapa)
        df = pd.read_csv('data.csv')    # tersimpan di /content/
        # atau tanpa tahu namanya:
        import io
        nama = list(uploaded)[0]
        df = pd.read_csv(io.BytesIO(uploaded[nama]))
    """, "File hilang saat runtime reset. Upload ulang dengan nama sama → jadi <code>data (1).csv</code>."),
    snip("Data bawaan (tanpa file)", """
        tips = sns.load_dataset('tips')        # juga: 'iris', 'titanic', 'penguins', 'flights'
        gap  = px.data.gapminder()             # juga: px.data.wind(), px.data.iris(), px.data.tips()
        df   = pd.read_csv('https://...csv')   # langsung dari URL
    """),
    snip("Cek data dulu", """
        df.shape; df.head(); df.info()
        df.describe()                          # numerik
        df.describe(include='object')          # kategorikal
        df['kolom'].value_counts()
        df.isna().sum()
    """),
])

# ---------------------------------------------------------------------------
PICK = """
<div class="tbl"><table>
<thead><tr><th>Pertanyaan</th><th>Kelompok</th><th>Grafik</th><th>Fungsi</th></tr></thead>
<tbody>
<tr><td>Bandingkan nilai antar kategori</td><td><span class="chip c-cat">categorical</span></td><td>bar, dot plot</td><td><code>sns.barplot</code> · <code>px.bar</code> · <code>sns.catplot</code></td></tr>
<tr><td>Hitung frekuensi tiap kategori</td><td><span class="chip c-cat">categorical</span></td><td>count plot</td><td><code>sns.countplot</code></td></tr>
<tr><td>Sebaran 1 variabel numerik</td><td><span class="chip c-cat">categorical</span></td><td>histogram, KDE</td><td><code>sns.histplot</code> · <code>sns.kdeplot</code></td></tr>
<tr><td>Sebaran + outlier per grup</td><td><span class="chip c-rel">relational</span></td><td>boxplot, violin</td><td><code>sns.boxplot</code> · <code>sns.violinplot</code></td></tr>
<tr><td>Arah mata angin / siklus</td><td><span class="chip c-cat">categorical</span></td><td>radial / polar bar</td><td><code>px.bar_polar</code></td></tr>
<tr><td>Aliran dari sumber ke tujuan</td><td><span class="chip c-cat">categorical</span></td><td>sankey</td><td><code>go.Sankey</code></td></tr>
<tr><td>Frekuensi kata</td><td><span class="chip c-cat">categorical</span></td><td>word cloud</td><td><code>WordCloud</code></td></tr>
<tr><td>Bagian dari keseluruhan</td><td><span class="chip c-hie">hierarchical</span></td><td>pie, waffle, stacked bar</td><td><code>plt.pie</code> · <code>Waffle</code> · <code>plot(stacked=True)</code></td></tr>
<tr><td>Hierarki bertingkat + ukuran</td><td><span class="chip c-hie">hierarchical</span></td><td>tree map</td><td><code>px.treemap</code></td></tr>
<tr><td>Tren terhadap waktu</td><td><span class="chip c-tem">temporal</span></td><td>line, area</td><td><code>df.plot</code> · <code>plt.stackplot</code></td></tr>
<tr><td>Harga open/high/low/close</td><td><span class="chip c-tem">temporal</span></td><td>candlestick</td><td><code>go.Candlestick</code></td></tr>
<tr><td>Hubungan 2 numerik</td><td><span class="chip c-rel">relational</span></td><td>scatter (+ hue)</td><td><code>sns.scatterplot</code> · <code>sns.regplot</code></td></tr>
<tr><td>Hubungan 3 numerik</td><td><span class="chip c-rel">relational</span></td><td>bubble</td><td><code>px.scatter(size=...)</code></td></tr>
<tr><td>Hubungan banyak numerik</td><td><span class="chip c-rel">relational</span></td><td>scatter matrix, heatmap korelasi</td><td><code>sns.pairplot</code> · <code>go.Splom</code> · <code>sns.heatmap</code></td></tr>
<tr><td>Data dengan lokasi</td><td><span class="chip c-spa">spatial</span></td><td>choropleth</td><td><code>px.choropleth</code></td></tr>
</tbody></table></div>
"""

RANK = """
<p class="lead-s">Paling akurat → paling tidak akurat untuk nilai <b>kuantitatif</b> (Spoerri):</p>
<ol class="rank"><li>Position</li><li>Length</li><li>Angle · Slope</li><li>Area</li><li>Volume</li><li>Color · Density</li></ol>
<div class="tbl"><table>
<thead><tr><th>Tipe data</th><th>Channel terbaik</th><th>Hindari</th></tr></thead>
<tbody>
<tr><td><b>Numerik</b></td><td>position, length, angle, slope</td><td>color hue</td></tr>
<tr><td><b>Ordinal</b></td><td>position, density, color saturation</td><td>color hue acak</td></tr>
<tr><td><b>Kategorikal</b></td><td>position, color hue, texture, shape</td><td>length, saturation</td></tr>
</tbody></table></div>
<p class="note">Hue (merah/hijau/biru) <b>tidak punya urutan alami</b>. Untuk nilai berurutan pakai gradasi terang–gelap (saturation/lightness), contoh <code>cmap='Blues'</code>.</p>
"""

section("pilih", "Pilih grafik", "", "Taksonomi Andy Kirk (2012): categorical · hierarchical · relational · temporal · spatial.", [
    rich("Pertanyaan → grafik", PICK),
    rich("Ranking visual encoding", RANK),
])

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bentuk data tiap grafik: mini tables computed with pandas on a tiny example
import pandas as pd

TRX = pd.DataFrame({
    "tanggal": ["01-09", "01-09", "02-09", "02-09", "02-09", "03-09"],
    "cabang": ["Bandung", "Jakarta", "Jakarta", "Bandung", "Jakarta", "Bandung"],
    "kategori": ["Kopi", "Kopi", "Makanan", "Kopi", "Kopi", "Makanan"],
    "total": [54, 27, 33, 81, 27, 66],
})


def mini(obj, caption=""):
    """Render a small DataFrame/Series as a compact HTML table (index shown in bold)."""
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    idx_name = df.index.name or ""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join(
        "<tr><th>" + html.escape(str(i)) + "</th>" + "".join(f'<td class="num">{html.escape(str(v))}</td>' if isinstance(v, (int, float)) and not isinstance(v, bool)
                                                              else f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
        for i, r in zip(df.index, df.itertuples(index=False)))
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tbl"><table class="mini">{cap}<thead><tr><th>{html.escape(idx_name)}</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def pattern(title, code, before_html, after_obj, after_cap, used_for):
    code = textwrap.dedent(code).strip("\n")
    all_snippets.append((title, code))
    body = (f'<div class="flowpair"><div>{before_html}</div><div class="arrow" aria-hidden="true">→</div>'
            f'<div>{mini(after_obj, after_cap)}</div></div>'
            f'<pre><code class="language-python">{html.escape(code)}</code></pre>'
            f'<p class="note"><b>Dipakai untuk:</b> {used_for}</p>')
    return {"kind": "html", "title": title, "body": body, "fig": None, "wide": False,
            "search": title + " " + code + " " + used_for}


raw_html = mini(TRX.set_index(pd.RangeIndex(len(TRX))), "data mentah: 1 baris = 1 transaksi (total dalam ribu Rp)")
raw_small = mini(TRX[["cabang", "total"]], "mentah")

SHAPE_TBL = """
<div class="tbl"><table class="shape">
<thead><tr><th>Grafik</th><th>Data yang dimasukkan</th><th>Perlu diringkas?</th><th>Cara menyiapkan</th></tr></thead>
<tbody>
<tr><td>Histogram, KDE, boxplot, violin</td><td>mentah: 1 kolom numerik (+ kategori untuk <code>hue</code>/grup)</td><td><span class="dk raw">tidak</span></td><td>langsung <code>data=df, x='kolom'</code></td></tr>
<tr><td>Count plot</td><td>mentah: 1 kolom kategori</td><td><span class="dk raw">tidak</span></td><td>fungsi menghitung sendiri</td></tr>
<tr><td>Scatter, regplot, pairplot</td><td>mentah: ≥ 2 kolom numerik, 1 baris = 1 titik</td><td><span class="dk raw">tidak</span></td><td>langsung; filter dulu bila perlu (<code>query</code>)</td></tr>
<tr><td>Bar chart (jumlah)</td><td>1 baris per kategori + jumlah</td><td><span class="dk agg">ya</span></td><td><code>value_counts()</code> · <code>groupby('kat').size()</code></td></tr>
<tr><td>Bar chart (total / rata-rata)</td><td>1 baris per kategori + nilai</td><td><span class="dk agg">ya</span></td><td><code>groupby('kat')['nilai'].sum()</code> / <code>.mean()</code></td></tr>
<tr><td>Pie, donut, waffle</td><td>1 baris per kategori + nilai</td><td><span class="dk agg">ya</span></td><td><code>value_counts()</code> · <code>groupby().sum()</code></td></tr>
<tr><td>Stacked bar, heatmap 2 kategori</td><td>tabel wide: baris = kategori A, kolom = kategori B</td><td><span class="dk agg">ya</span></td><td><code>pd.crosstab</code> · <code>groupby([a, b]).size().unstack()</code> · <code>pivot_table</code></td></tr>
<tr><td>Line / area (1 seri)</td><td>1 nilai per tanggal, terurut</td><td><span class="dk agg">ya, jika banyak baris per tanggal</span></td><td><code>groupby('tanggal').sum()</code> · <code>resample('D')</code></td></tr>
<tr><td>Line (banyak seri)</td><td>wide (<code>df.plot</code>) atau long + kolom seri (<code>hue</code>)</td><td><span class="dk agg">ya</span></td><td><code>pivot_table</code> (→ wide) · <code>melt</code> (→ long)</td></tr>
<tr><td>Candlestick</td><td>1 baris per periode: Open, High, Low, Close</td><td><span class="dk agg">ya, jika ganti periode</span></td><td><code>resample('W').agg({...})</code></td></tr>
<tr><td>Heatmap korelasi</td><td>matriks kolom × kolom</td><td><span class="dk agg">ya</span></td><td><code>df.corr(numeric_only=True)</code></td></tr>
<tr><td>Tree map</td><td>1 baris per daun + kolom tiap level + nilai</td><td><span class="dk raw">tidak wajib</span></td><td>plotly menjumlahkan per <code>path</code></td></tr>
<tr><td>Sankey</td><td>link: asal, tujuan, nilai</td><td><span class="dk agg">ya</span></td><td><code>groupby(['asal', 'tujuan']).size()</code> + indeks node</td></tr>
<tr><td>Word cloud</td><td>satu string panjang / dict frekuensi</td><td><span class="dk agg">gabung teks</span></td><td><code>' '.join(df['kolom'])</code></td></tr>
</tbody></table></div>
<p class="note"><b>Aturan praktis:</b> grafik yang menggambar <b>sebaran atau tiap titik</b> (histogram, boxplot, scatter) memakai data <b>mentah</b>. Grafik yang menggambar <b>satu nilai per kategori/tanggal</b> (bar, pie, line, stacked) membutuhkan data yang sudah <b>diringkas</b>. Kalau bar chart-mu hanya berisi satu batang setinggi jumlah baris, atau line chart-mu zig-zag tidak karuan, biasanya datanya belum di-<code>groupby</code>.</p>
"""

section("bentuk", "Bentuk data tiap grafik", "", "Data mentah atau diringkas dulu? Contoh di bawah memakai 6 transaksi yang sama.", [
    rich("Grafik → bentuk data → cara menyiapkan", SHAPE_TBL),
    rich("Contoh data mentah", raw_html + '<p class="note">Setiap kartu di bawah mengubah tabel ini menjadi bentuk yang dibutuhkan grafik tertentu.</p>'),
    pattern("Hitung baris per kategori", """
        jumlah = df['cabang'].value_counts()
        # atau: df.groupby('cabang').size()
        jumlah.plot(kind='bar')
    """, raw_small, TRX["cabang"].value_counts().rename_axis("cabang"), "<code>value_counts()</code>",
        "bar chart jumlah, pie, waffle. (<code>sns.countplot</code> tidak perlu langkah ini.)"),
    pattern("Total / rata-rata per kategori", """
        ringkas = df.groupby('cabang')['total'].agg(['sum', 'mean'])
        ringkas['sum'].plot(kind='barh')
    """, raw_small, TRX.groupby("cabang")["total"].agg(["sum", "mean"]).round(1), "<code>groupby().agg()</code>",
        "bar chart nilai (total, rata-rata), pie berdasarkan nilai."),
    pattern("Dua kategori → tabel silang", """
        tabel = pd.crosstab(df['cabang'], df['kategori'])
        # sama: df.groupby(['cabang', 'kategori']).size().unstack(fill_value=0)
        tabel.plot(kind='bar', stacked=True)
    """, mini(TRX[["cabang", "kategori"]], "mentah"), pd.crosstab(TRX["cabang"], TRX["kategori"]), "<code>crosstab</code> (wide)",
        "stacked bar, grouped bar, heatmap. Pakai <code>normalize='index'</code> untuk 100% stacked."),
    pattern("Transaksi → deret waktu", """
        harian = df.groupby('tanggal')['total'].sum()
        # kolom datetime: df.set_index('tanggal')['total'].resample('D').sum()
        harian.plot(marker='o')
    """, mini(TRX[["tanggal", "total"]], "mentah"), TRX.groupby("tanggal")["total"].sum(), "1 nilai per tanggal",
        "line chart, area chart, moving average. Tanpa langkah ini, line chart menghubungkan tiap transaksi sehingga zig-zag."),
    pattern("Long → wide (pivot)", """
        wide = df.pivot_table(index='tanggal', columns='cabang',
                              values='total', aggfunc='sum', fill_value=0)
        wide.plot()                      # satu garis per kolom
    """, mini(TRX[["tanggal", "cabang", "total"]], "long"),
        TRX.pivot_table(index="tanggal", columns="cabang", values="total", aggfunc="sum", fill_value=0), "wide",
        "<code>df.plot()</code> banyak seri, <code>plt.stackplot</code>, heatmap. pandas/matplotlib membaca <b>satu kolom = satu seri</b>."),
    pattern("Wide → long (melt)", """
        long = wide.reset_index().melt(id_vars='tanggal', var_name='cabang',
                                       value_name='total')
        sns.lineplot(data=long, x='tanggal', y='total', hue='cabang')
    """, mini(TRX.pivot_table(index="tanggal", columns="cabang", values="total", aggfunc="sum", fill_value=0), "wide"),
        TRX.pivot_table(index="tanggal", columns="cabang", values="total", aggfunc="sum", fill_value=0).reset_index()
           .melt(id_vars="tanggal", var_name="cabang", value_name="total").set_index("tanggal"), "long",
        "seaborn & plotly (<code>hue=</code>, <code>color=</code>): <b>satu kolom berisi nama seri</b>."),
])

section("prep", "Siapkan data untuk plot", "", "Kebanyakan grafik butuh data yang sudah diringkas. Pola yang paling sering dipakai:", [
    snip("Hitung jumlah per kategori", """
        cnt = df.groupby('province').size().to_frame('count').reset_index()
        cnt = df['province'].value_counts().reset_index()      # kolom: province, count
        cnt = cnt.sort_values('count', ascending=False)
    """),
    snip("Agregasi (mean, sum, …)", """
        df.groupby('kategori')['nilai'].mean().reset_index()
        df.groupby('kategori').agg(rata=('nilai', 'mean'), total=('nilai', 'sum'))
    """),
    snip("Tabel 2 kategori (untuk stacked bar / heatmap)", """
        tmp = df.groupby(['province', 'gender']).size().unstack(fill_value=0)
        tmp = pd.crosstab(df['province'], df['gender'])                     # sama hasilnya
        pv  = df.pivot_table(index='bulan', columns='tahun', values='nilai', aggfunc='sum')
    """, "Kalau unstack menghasilkan kolom bertingkat: <code>tmp.columns = tmp.columns.droplevel(0)</code>."),
    snip("Wide ↔ long", """
        long = df.melt(id_vars='date', value_vars=['a', 'b'], var_name='status', value_name='jumlah')
        wide = long.pivot(index='date', columns='status', values='jumlah')
    """, "seaborn & plotly paling enak dengan format <b>long</b> (satu kolom kategori untuk <code>hue</code>/<code>color</code>)."),
    snip("Tanggal", """
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        bulanan = df.set_index('date')['nilai'].resample('MS').sum()   # 'W' minggu, 'MS' bulan, 'YS' tahun
        df['tahun'] = df['date'].dt.year
    """),
    snip("Binning numerik → kategori", """
        df['kelompok_usia'] = pd.cut(df['age'], bins=[0, 18, 40, 60, 100],
                                     labels=['anak', 'dewasa muda', 'dewasa', 'lansia'])
        df['kuartil'] = pd.qcut(df['charges'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    """),
    snip("Filter & ambil top-N", """
        df2007 = df.query('year == 2007')
        top10 = cnt.nlargest(10, 'count')
        df = df.dropna(subset=['age'])
    """),
])

# ---------------------------------------------------------------------------
section("anatomi", "Anatomi grafik matplotlib", "", "Semua seaborn & pandas plot berdiri di atas matplotlib, jadi perintah ini berlaku untuk keduanya.", [
    snip("Kerangka satu grafik", """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cnt, x='count', y='province', ax=ax)
        ax.set_title('Judul Grafik', fontsize=16, weight='bold')
        ax.set_xlabel('jumlah kasus'); ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='gender', loc='upper right')   # atau bbox_to_anchor=(1.02, 1)
        plt.tight_layout()
        plt.show()
    """),
    snip("Subplot (small multiples)", """
        fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axs[0, 0].plot(x, y1); axs[0, 0].set_title('A')
        axs[0, 1].plot(x, y2); axs[0, 1].set_title('B')
        for ax in axs.flat:
            ax.set(xlabel='tanggal', ylabel='kasus')
        fig.suptitle('Judul Besar', fontsize=16)
        fig.tight_layout(); plt.show()
    """),
    snip("Gaya plt (tanpa ax)", """
        plt.figure(figsize=(10, 6))
        plt.title('Judul'); plt.xlabel('x'); plt.ylabel('y')
        plt.xticks(rotation=45); plt.ylim(0, None); plt.grid(alpha=.3)
        plt.legend(loc='upper left')
        plt.savefig('grafik.png', dpi=150, bbox_inches='tight')
        plt.show()
    """, "Jangan panggil <code>plt.figure()</code> sebelum <code>df.plot(figsize=...)</code>: muncul figure kosong."),
    snip("Warna & label nilai", """
        sns.barplot(data=df, x='day', y='tip', hue='day', palette='viridis', legend=False)
        # palette lain: 'Set2', 'pastel', 'colorblind', 'Blues', 'RdBu' (seaborn 0.13: palette butuh hue)
        sns.barplot(..., color='steelblue')       # satu warna saja
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, n)]
        ax.bar_label(ax.containers[0], fmt='%.0f')   # angka di atas batang
    """),
])

# ---------------------------------------------------------------------------
section("dist", "Distribusi", "c-cat", "Materi 12a: histogram, count plot, boxplot.", [
    snip("Histogram (numerik)", """
        sns.histplot(data=df, x='age', bins=20)
        sns.histplot(data=df, x='age', kde=True)               # + garis kepadatan
        sns.histplot(data=df, x='age', hue='smoker', element='step')
        plt.hist(df['age'].dropna(), bins=20, density=True)      # versi matplotlib
    """, "<code>sns.distplot</code> sudah deprecated → pakai <code>histplot</code> (atau <code>displot</code> untuk figure-level). <code>normed=True</code> → <code>density=True</code>."),
    snip("KDE", """
        sns.kdeplot(data=df, x='bmi', hue='smoker', fill=True)
    """),
    snip("Count plot (kategorikal)", """
        sns.countplot(data=df, x='day')
        sns.countplot(data=df, y='province', order=df['province'].value_counts().index)
        sns.countplot(data=df, x='day', hue='sex')
    """, "Count plot = histogram untuk kategori: tinggi batang = jumlah baris."),
    snip("Boxplot", """
        sns.boxplot(data=df, y='charges')                         # 1 variabel
        sns.boxplot(data=df, x='smoker', y='charges')             # per grup
        sns.boxplot(data=df, x='day', y='total_bill', hue='sex')  # 2 kategori
    """),
    snip("Violin / strip", """
        sns.violinplot(data=df, x='smoker', y='charges')
        sns.stripplot(data=df, x='smoker', y='charges', alpha=.4)
    """),
    snip("Dot plot (catplot)", """
        sns.catplot(data=pasien, x='age', y='province', hue='current_state',
                    height=6, aspect=1.5)                  # kind='strip' (default), 'box', 'bar', 'count'
        plt.title('Judul', weight='bold'); plt.show()
    """),
])

# ---------------------------------------------------------------------------
section("kategori", "Categorical", "c-cat", "Membandingkan ukuran relatif & absolut antar kategori.", [
    snip("Bar chart · seaborn", """
        sns.barplot(data=cnt, x='count', y='province', color='steelblue')
        sns.barplot(data=df, x='day', y='total_bill', errorbar=None)    # mean per kategori
    """, "<code>barplot</code> menghitung <b>mean</b> + error bar. Untuk jumlah baris pakai <code>countplot</code>."),
    snip("Bar chart · plotly", """
        fig = px.bar(cnt, x='province', y='count', text='count')
        fig.update_traces(textposition='outside')
        fig.update_layout(title='Jumlah Kasus<br>per Provinsi', width=700)
        fig.show()
    """, "Ganti baris di judul plotly pakai <code>&lt;br&gt;</code>, bukan <code>\\n</code>."),
    snip("Radial / polar bar", """
        df = px.data.wind()
        fig = px.bar_polar(df, r='frequency', theta='direction', color='strength',
                           color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig.show()
    """),
    snip("Sankey", """
        fig = go.Figure(go.Sankey(
            node=dict(label=['A', 'B', 'X', 'Y'], pad=15, thickness=15),
            link=dict(source=[0, 0, 1], target=[2, 3, 3], value=[8, 2, 5])))
        fig.update_layout(title_text='Aliran'); fig.show()
    """, "<code>source</code>/<code>target</code> berisi <b>indeks</b> pada list <code>label</code>."),
    snip("Word cloud", """
        from wordcloud import WordCloud, STOPWORDS
        teks = ' '.join(df['title'].dropna().astype(str))      # BUKAN str(df['title'])
        wc = WordCloud(background_color='white', stopwords=set(STOPWORDS),
                       width=800, height=400).generate(teks)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear'); plt.axis('off'); plt.show()
    """),
])

# ---------------------------------------------------------------------------
section("hier", "Hierarchical", "c-hie", "Bagian terhadap keseluruhan (part-to-whole).", [
    snip("Pie / donut", """
        s = df['province'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(s.values, labels=s.index, autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.4))                    # hapus wedgeprops → pie biasa
        plt.title('Komposisi'); plt.show()
        px.pie(cnt, names='province', values='count', hole=.4).show()   # versi plotly
    """, "Maksimal ±5 irisan. Lebih dari itu, bar chart terurut lebih mudah dibaca."),
    snip("Stacked bar", """
        tmp = pd.crosstab(df['province'], df['gender'])
        tmp.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Kasus per Provinsi & Gender'); plt.show()
        # persentase (100% stacked):
        tmp.div(tmp.sum(axis=1), axis=0).plot(kind='barh', stacked=True)
    """),
    snip("Waffle (square pie)", """
        !pip install -q pywaffle
        from pywaffle import Waffle
        d = df['Fuel_Type'].value_counts()
        fig = plt.figure(FigureClass=Waffle, rows=7, figsize=(10, 6),
                         values=d.to_dict(),                 # {label: jumlah}
                         legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)},
                         title={'label': 'Kendaraan per Bahan Bakar'})
        plt.show()
    """),
    snip("Tree map", """
        df = px.data.gapminder().query('year == 2007')
        fig = px.treemap(df, path=[px.Constant('world'), 'continent', 'country'],
                         values='pop', color='lifeExp', color_continuous_scale='RdBu',
                         color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))
        fig.show()
    """, "Luas = <code>values</code>, warna = variabel kedua. <code>path</code> berurutan dari akar ke daun."),
])

# ---------------------------------------------------------------------------
section("temporal", "Temporal", "c-tem", "Perubahan dan tren sepanjang waktu. Ubah kolom tanggal ke datetime dulu.", [
    snip("Line chart", """
        cases.plot(x='date', y=['acc_tested', 'acc_confirmed'], figsize=(10, 6))
        sns.lineplot(data=long, x='date', y='jumlah', hue='status')    # format long
        px.line(long, x='date', y='jumlah', color='status').show()
    """, "Sumbu y line chart <b>boleh</b> tidak mulai dari 0 (yang dibaca adalah pola)."),
    snip("Area & stacked area", """
        plt.stackplot(cases['date'], cases['a'], cases['b'],
                      labels=['a', 'b'], colors=['orange', 'green'], alpha=.6)
        plt.legend(loc='upper left'); plt.xticks(rotation=45); plt.show()
        cases.plot.area(x='date', y=['a', 'b'], alpha=.6)               # versi pandas
    """, "Area chart <b>wajib</b> mulai dari 0. Stacked hanya untuk bagian yang saling lepas (jangan tumpuk total + bagiannya)."),
    snip("Candlestick", """
        fig = go.Figure(go.Candlestick(x=data['Date'], open=data['Open'],
                        high=data['High'], low=data['Low'], close=data['Close']))
        fig.update_layout(title_text='Harga Saham', xaxis_rangeslider_visible=True)
        fig.show()
    """, "Body = open–close, sumbu = low–high. Hijau naik (close &gt; open), merah turun."),
    snip("Moving average", """
        s = df.set_index('date')['Close']
        s.plot(alpha=.4, label='harian')
        s.rolling(30).mean().plot(label='MA-30')
        plt.legend(); plt.show()
    """),
])

# ---------------------------------------------------------------------------
section("relasi", "Relational", "c-rel", "Hubungan, korelasi, klaster, dan outlier antar variabel.", [
    snip("Scatter", """
        sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', style='sex', alpha=.7)
        sns.regplot(data=df, x='bmi', y='charges', scatter_kws={'alpha': .4})   # + garis regresi
        sns.lmplot(data=df, x='bmi', y='charges', hue='smoker')                  # regresi per grup
    """),
    snip("Korelasi + heatmap", """
        corr = df.corr(numeric_only=True)              # numeric_only wajib di pandas 2
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Matriks Korelasi'); plt.show()
        df['bmi'].corr(df['charges'])                  # satu pasangan
    """, "Pakai colormap <b>diverging</b> (<code>coolwarm</code>, <code>RdBu</code>) dengan <code>vmin=-1, vmax=1</code> agar 0 berada di tengah."),
    snip("Heatmap pivot", """
        pv = sns.load_dataset('flights').pivot(index='month', columns='year', values='passengers')
        sns.heatmap(pv, cmap='YlGnBu', annot=True, fmt='d')
    """),
    snip("Bubble", """
        fig = px.scatter(gap.query('year == 2007'), x='gdpPercap', y='lifeExp',
                         size='pop', color='continent', hover_name='country',
                         size_max=60, log_x=True)
        fig.show()
    """),
    snip("Scatter matrix", """
        sns.pairplot(df, vars=['Glucose', 'BMI', 'Age'], hue='Outcome', corner=True)
        px.scatter_matrix(df, dimensions=['Glucose', 'BMI', 'Age'], color='Outcome').show()
    """),
])

# ---------------------------------------------------------------------------
ERR = """
<div class="tbl"><table>
<thead><tr><th>Gejala / error</th><th>Penyebab</th><th>Perbaikan</th></tr></thead>
<tbody>
<tr><td><code>FileNotFoundError</code></td><td>path salah, file belum di-upload, runtime reset</td><td>cek <code>!ls "/content/drive/My Drive/..."</code>; perhatikan spasi & huruf besar</td></tr>
<tr><td><code>KeyError: 'Date'</code></td><td>nama kolom beda (spasi/huruf kecil)</td><td><code>df.columns = df.columns.str.strip()</code>, cek <code>df.columns</code></td></tr>
<tr><td><code>could not convert string to float</code> di <code>corr()</code></td><td>ada kolom teks</td><td><code>df.corr(numeric_only=True)</code></td></tr>
<tr><td>Warning <code>distplot is deprecated</code></td><td>seaborn ≥ 0.11</td><td><code>sns.histplot</code> / <code>sns.displot</code></td></tr>
<tr><td><code>unexpected keyword 'normed'</code></td><td>dihapus dari matplotlib</td><td><code>density=True</code></td></tr>
<tr><td>Muncul figure kosong</td><td><code>plt.figure()</code> lalu <code>df.plot(figsize=…)</code></td><td>hapus <code>plt.figure()</code>, atau kirim <code>ax=</code></td></tr>
<tr><td>Judul/label ganda tertimpa</td><td>tidak ada jarak</td><td><code>plt.tight_layout()</code></td></tr>
<tr><td>Label sumbu x bertumpuk</td><td>kategori banyak / tanggal</td><td><code>plt.xticks(rotation=45, ha='right')</code> atau pakai bar horizontal</td></tr>
<tr><td>Urutan tanggal acak</td><td>kolom tanggal masih string</td><td><code>pd.to_datetime</code> lalu <code>sort_values</code></td></tr>
<tr><td>Histogram error karena NaN</td><td><code>plt.hist</code> tidak terima NaN</td><td><code>.dropna()</code></td></tr>
<tr><td>Word cloud cuma berisi ±10 judul + "Name", "dtype"</td><td><code>str(series)</code> mengambil tampilan cetak</td><td><code>' '.join(series.astype(str))</code></td></tr>
<tr><td>Kolom hasil <code>unstack</code> bertingkat</td><td>MultiIndex kolom</td><td><code>.columns.droplevel(0)</code> atau <code>pd.crosstab</code></td></tr>
<tr><td>Grafik plotly tidak muncul</td><td>renderer</td><td><code>import plotly.io as pio; pio.renderers.default = 'colab'</code></td></tr>
<tr><td><code>ModuleNotFoundError: pywaffle</code></td><td>belum ter-install</td><td><code>!pip install -q pywaffle</code></td></tr>
<tr><td>Warna berulang di pie</td><td>jumlah warna &lt; jumlah kategori</td><td><code>np.linspace(0, 1, len(data))</code></td></tr>
</tbody></table></div>
"""
section("error", "Error umum", "", "Yang paling sering muncul saat menjalankan kode visualisasi di Colab.", [
    rich("Gejala → perbaikan", ERR),
])

# ---------------------------------------------------------------------------
READ = """
<div class="grid2">
<div>
<h4>Distribusi (histogram/KDE)</h4>
<ul>
<li><b>Bentuk:</b> simetris, miring kanan (ekor ke kanan, mean &gt; median), miring kiri (mean &lt; median).</li>
<li><b>Puncak:</b> unimodal / bimodal (dua kelompok berbeda).</li>
<li><b>Pusat & rentang:</b> nilai yang paling sering, min–max.</li>
<li><b>Outlier:</b> batang terpisah jauh dari kelompok utama.</li>
</ul>
</div>
<div>
<h4>Boxplot</h4>
<ul>
<li>Kotak = Q1–Q3 (IQR, 50% data tengah), garis = median.</li>
<li>Whisker sampai titik terjauh dalam 1,5 × IQR.</li>
<li>Titik di luar whisker = outlier (&lt; Q1 − 1,5·IQR atau &gt; Q3 + 1,5·IQR).</li>
<li>Median tidak di tengah kotak → distribusi miring.</li>
</ul>
</div>
<div>
<h4>Korelasi (r)</h4>
<div class="tbl"><table class="compact">
<thead><tr><th>|r|</th><th>Kekuatan</th></tr></thead>
<tbody>
<tr><td class="num">0,00–0,19</td><td>sangat lemah</td></tr>
<tr><td class="num">0,20–0,39</td><td>lemah</td></tr>
<tr><td class="num">0,40–0,59</td><td>sedang</td></tr>
<tr><td class="num">0,60–0,79</td><td>kuat</td></tr>
<tr><td class="num">0,80–1,00</td><td>sangat kuat</td></tr>
</tbody></table></div>
<p class="note">Tanda + : naik bersama. Tanda − : satu naik, satu turun. Korelasi ≠ sebab-akibat. r hanya mengukur hubungan <b>linear</b> (ingat Anscombe's quartet).</p>
</div>
<div>
<h4>Tren waktu</h4>
<ul>
<li><b>Arah:</b> naik / turun / datar.</li>
<li><b>Laju:</b> kapan paling curam (percepatan).</li>
<li><b>Fluktuasi & musiman:</b> pola berulang tiap minggu/bulan/tahun.</li>
<li><b>Titik penting:</b> puncak, lembah, perpotongan dua garis.</li>
</ul>
</div>
</div>
<h4>Template kalimat interpretasi</h4>
<ul class="tpl">
<li>Grafik <i>[jenis]</i> menunjukkan <i>[variabel]</i> berdasarkan <i>[kategori/waktu]</i>.</li>
<li><i>[Kategori A]</i> memiliki nilai tertinggi (<i>[angka]</i>), diikuti <i>[B]</i> dan <i>[C]</i>; terendah adalah <i>[Z]</i>.</li>
<li>Distribusi <i>[variabel]</i> miring ke kanan: sebagian besar data berada di <i>[rentang]</i>, dengan sedikit nilai ekstrem hingga <i>[angka]</i>.</li>
<li>Terdapat korelasi <i>[positif/negatif]</i> <i>[kekuatan]</i> antara <i>[X]</i> dan <i>[Y]</i> (r = <i>[angka]</i>).</li>
<li>Kelompok <i>[hue]</i> membentuk klaster terpisah, sehingga <i>[hue]</i> berpengaruh pada <i>[Y]</i>.</li>
<li><b>Kesimpulan/insight:</b> <i>[apa artinya bagi tujuan analisis]</i>. <b>Saran:</b> <i>[tindakan atau analisis lanjutan]</i>.</li>
</ul>
"""
section("baca", "Membaca & menulis interpretasi", "", "Soal analisis biasanya meminta: apa yang terlihat, angka pendukung, dan artinya.", [
    rich("Cara membaca", READ),
])

# ---------------------------------------------------------------------------
PRINC = """
<div class="grid2">
<div>
<h4>Tujuan visualisasi</h4>
<ul>
<li><b>Exploratory</b> (untuk analisis): coba banyak tampilan untuk menemukan pola.</li>
<li><b>Explanatory</b> (untuk komunikasi): satu tampilan terpilih untuk menyampaikan pesan.</li>
<li><b>Exhibition</b>: visualisasi sebagai karya/pameran.</li>
<li>Tone: <b>pragmatic/analytic</b> (cepat & akurat dibaca) vs <b>emotive</b>.</li>
</ul>
<h4>Teknik visual analitik</h4>
<ul>
<li><b>Comparison & proportion:</b> range, distribution, ranking, measurement.</li>
<li><b>Trend & pattern:</b> direction, rate of change, fluctuation, intersection.</li>
<li><b>Relationship:</b> exceptions, correlation, association, cluster & gap, hierarchy.</li>
</ul>
</div>
<div>
<h4>Aturan desain</h4>
<ul>
<li>Tentukan pesan/pertanyaan dulu, baru pilih bentuk grafik.</li>
<li>Bar & area: sumbu y mulai dari <b>0</b>. Line boleh tidak.</li>
<li>Yang dibandingkan harus berdekatan dan memakai skala sama.</li>
<li>Konsisten: yang sama tampak sama, yang beda tampak beda (warna kategori tetap di semua grafik).</li>
<li><b>Redundant encoding</b>: satu variabel di dua channel (mis. warna + bentuk) agar lebih cepat dibaca.</li>
<li>Pakai bentuk standar kecuali bentuk baru benar-benar lebih baik bagi pembaca.</li>
<li>Maksimalkan data-ink ratio; hindari 3D, bayangan, dan dekorasi.</li>
<li>Selalu ada judul, label sumbu + satuan, dan legend.</li>
<li>Ramah buta warna: hindari merah–hijau saja; <code>palette='colorblind'</code>.</li>
</ul>
</div>
</div>
"""
section("prinsip", "Prinsip desain (12a)", "", "Bahan untuk soal teori atau alasan pemilihan grafik.", [
    rich("Ringkasan", PRINC),
])

# ===========================================================================
# TAB 2: CLEANING DATA
CUR_TAB[0] = "clean"

ALUR = """
<ol class="steps">
<li><b>Diagnosis</b><span><code>info()</code>, <code>describe()</code>, <code>isna().sum()</code>, cari placeholder (<code>"?"</code>, <code>"-"</code>, spasi)</span></li>
<li><b>Seragamkan</b><span>nama kolom, spasi & huruf besar pada teks kategori</span></li>
<li><b>Perbaiki tipe</b><span>placeholder → <code>NaN</code>, teks angka → numerik, tanggal → datetime</span></li>
<li><b>Duplikat</b><span>hitung lalu hapus <i>sebelum</i> imputasi agar statistik tidak bias</span></li>
<li><b>Missing values</b><span>buang atau isi (median / mean / modus / ffill)</span></li>
<li><b>Outlier</b><span>deteksi (IQR / z-score), lalu buang, capping, atau pertahankan sesuai konteks</span></li>
<li><b>Buang kolom</b><span>konstan, ID, missing &gt; ±50%, redundan</span></li>
<li><b>Encoding & scaling</b><span>ordinal → <code>map</code>, nominal → one-hot, numerik → Min-Max / z-score</span></li>
<li><b>Validasi</b><span>tidak ada NaN & duplikat, rentang masuk akal, laporkan ukuran sebelum → sesudah</span></li>
</ol>
<p class="note">Urutan ini penting: <code>"?"</code> belum terhitung sebagai NaN sebelum tipe diperbaiki, dan <code>"Jakarta "</code> vs <code>"jakarta"</code> tidak terdeteksi duplikat sebelum teks diseragamkan.</p>
"""
section("c-alur", "Alur cleaning", "", "Kerjakan berurutan. Setiap langkah, tampilkan bukti sebelum & sesudah.", [
    rich("Langkah", ALUR),
])

section("c-diag", "1 · Diagnosis kualitas data", "", "Temukan masalahnya dulu sebelum memperbaiki.", [
    snip("Ringkasan cepat", """
        print(df.shape)
        df.info()                                  # dtype janggal? angka terbaca 'object'?
        df.describe(include='all').T               # min/max aneh, unique, top
        df.nunique()                               # kolom konstan (1 nilai) / ID (semua unik)
    """),
    snip("Missing per kolom (jumlah & %)", """
        miss = pd.DataFrame({'jumlah': df.isna().sum(),
                             'persen': (df.isna().mean() * 100).round(1)})
        miss.sort_values('persen', ascending=False)
        # grafik
        (df.isna().mean() * 100).sort_values().plot.barh()
        plt.xlabel('% missing'); plt.show()
    """, fig="c_missing"),
    snip("Placeholder tersembunyi", """
        for kol in df.select_dtypes('object'):
            print(kol, df[kol].unique()[:15])      # lihat '?', '-', 'NA', '', ' '
        (df == '?').sum()                          # jumlah '?' per kolom
        df['kota'].str.strip().eq('').sum()        # string kosong / spasi
    """, "<code>isna()</code> <b>tidak</b> menghitung <code>\"?\"</code> atau string kosong sebagai missing."),
    snip("Nilai tidak masuk akal", """
        df[(df['age'] < 0) | (df['age'] > 120)]
        df[df['price'] <= 0]
        df[~df['gender'].isin(['male', 'female'])]
        df['date'].min(), df['date'].max()         # tanggal di masa depan?
    """),
])

section("c-teks", "2 · Seragamkan kolom & teks", "", "Kategori yang sama harus ditulis sama persis.", [
    snip("Nama kolom", """
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True))
        df = df.rename(columns={'harga_rp': 'harga', 'tgl': 'tanggal'})
    """),
    snip("Teks kategori", """
        df['kota'] = df['kota'].str.strip().str.title()             # ' jakarta ' -> 'Jakarta'
        df['gender'] = (df['gender'].str.strip().str.lower()
                        .replace({'m': 'male', 'l': 'male', 'pria': 'male',
                                  'f': 'female', 'p': 'female', 'wanita': 'female'}))
        df['kota'].value_counts()                                   # cek hasilnya
    """),
    snip("Angka yang tersimpan sebagai teks", """
        # 'Rp 1.250.000' -> 1250000   (titik = pemisah ribuan)
        df['harga'] = pd.to_numeric(df['harga'].str.replace(r'[^0-9]', '', regex=True), errors='coerce')
        # '$1,200.50' -> 1200.50      (koma = ribuan, titik = desimal)
        df['usd'] = pd.to_numeric(df['usd'].str.replace(r'[$,]', '', regex=True), errors='coerce')
        # '75 kg' -> 75
        df['berat'] = df['berat'].str.extract(r'(\\d+\\.?\\d*)', expand=False).astype(float)
    """),
])

section("c-tipe", "3 · Placeholder & tipe data", "", "Ubah semua penanda kosong menjadi NaN, lalu perbaiki dtype.", [
    snip("Placeholder → NaN", """
        df = pd.read_csv('data.csv', na_values=['?', '-', 'NA', 'n/a', ''])   # saat membaca
        df = df.replace(['?', '-', 'NA', 'n/a', ''], np.nan)                 # setelah dibaca
        df = df.replace(r'^\\s*$', np.nan, regex=True)                        # string berisi spasi saja
    """),
    snip("Teks → numerik", """
        df['monthly_spend'] = pd.to_numeric(df['monthly_spend'], errors='coerce')   # '?' -> NaN
        df['monthly_spend'].dtype                                                   # float64
    """, "<code>errors='coerce'</code>: nilai yang tidak bisa dibaca jadi NaN, bukan error."),
    snip("Tanggal", """
        df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')
        df['tgl'] = pd.to_datetime(df['tgl'], format='%d/%m/%Y')     # format Indonesia
        df['tgl'] = pd.to_datetime(df['tgl'], dayfirst=True)         # alternatif
    """),
    snip("Tipe lain", """
        df['plan'] = df['plan'].astype('category')
        df['age'] = df['age'].round().astype('Int64')       # int yang boleh NaN (huruf I besar)
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
        df['id'] = df['id'].astype(str)                     # ID bukan angka untuk dihitung
    """),
])

section("c-dup", "4 · Duplikat", "", "", [
    snip("Cek & hapus duplikat", """
        print('Duplikat:', df.duplicated().sum())
        df[df.duplicated(keep=False)]                         # lihat semua salinannya
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.drop_duplicates(subset=['customer_id'], keep='last')   # unik per ID
    """, "Duplikat menggandakan bobot satu sampel sehingga rata-rata dan proporsi bias."),
])

MISS_TBL = """
<div class="tbl"><table>
<thead><tr><th>Kondisi</th><th>Tindakan</th><th>Kode</th></tr></thead>
<tbody>
<tr><td>Missing sedikit (&lt; ±5%) dan acak</td><td>buang baris</td><td><code>dropna(subset=[...])</code></td></tr>
<tr><td>Kolom target / kunci kosong</td><td>buang baris</td><td><code>dropna(subset=['target'])</code></td></tr>
<tr><td>Kolom missing &gt; ±50–60%</td><td>buang kolom</td><td><code>drop(columns=[...])</code></td></tr>
<tr><td>Numerik, miring atau ada outlier</td><td>isi <b>median</b></td><td><code>fillna(s.median())</code></td></tr>
<tr><td>Numerik, simetris (± normal)</td><td>isi mean</td><td><code>fillna(s.mean())</code></td></tr>
<tr><td>Kategorikal</td><td>isi modus atau <code>'Unknown'</code></td><td><code>fillna(s.mode()[0])</code></td></tr>
<tr><td>Nilai bergantung grup</td><td>median per grup</td><td><code>groupby().transform('median')</code></td></tr>
<tr><td>Deret waktu</td><td>ffill / interpolasi</td><td><code>ffill()</code> · <code>interpolate()</code></td></tr>
</tbody></table></div>
"""
section("c-miss", "5 · Missing values", "", "Pilih strategi berdasarkan jumlah missing, tipe kolom, dan bentuk distribusinya.", [
    rich("Strategi", MISS_TBL),
    snip("Buang", """
        df = df.dropna(subset=['age', 'churn'])              # baris yang kolom pentingnya kosong
        df = df.dropna(axis=1, thresh=int(0.5 * len(df)))    # simpan kolom yg terisi >= 50%
        df = df.drop(columns=['kolom_hampir_kosong'])
    """, "<code>dropna()</code> tanpa <code>subset</code> membuang baris yang kosong di kolom <i>mana pun</i>; bisa kehilangan banyak data."),
    snip("Isi numerik & kategorikal", """
        df['age'] = df['age'].fillna(df['age'].median())
        df['city'] = df['city'].fillna(df['city'].mode()[0])     # mode() mengembalikan Series
        num = df.select_dtypes('number').columns                 # banyak kolom sekaligus
        df[num] = df[num].fillna(df[num].median())
    """, "Pakai <b>assignment</b> (<code>df['a'] = df['a'].fillna(...)</code>). <code>df['a'].fillna(..., inplace=True)</code> tidak mengubah <code>df</code> di pandas 3."),
    snip("Median per grup", """
        df['age'] = df['age'].fillna(df.groupby('plan')['age'].transform('median'))
    """),
    snip("Tandai dulu, baru isi", """
        df['age_missing'] = df['age'].isna().astype(int)    # informasi 'pernah kosong' tidak hilang
        df['age'] = df['age'].fillna(df['age'].median())
    """),
    snip("Deret waktu", """
        df = df.sort_values('date')
        df['suhu'] = df['suhu'].ffill()                     # pakai nilai sebelumnya
        df['suhu'] = df['suhu'].interpolate()               # tarik garis lurus antar titik
    """, fig="c_ts"),
])

OUT_TBL = """
<div class="tbl"><table>
<thead><tr><th>Penyebab outlier</th><th>Contoh</th><th>Tindakan</th></tr></thead>
<tbody>
<tr><td>Galat input / mustahil</td><td>umur 999, harga negatif</td><td>buang baris, atau jadikan NaN lalu imputasi</td></tr>
<tr><td>Nilai ekstrem tapi nyata</td><td>pengguna berat, rumah mewah</td><td>pertahankan; capping atau transformasi log bila mengganggu model</td></tr>
<tr><td>Beda populasi</td><td>akun bisnis di data pelanggan pribadi</td><td>pisahkan analisisnya</td></tr>
</tbody></table></div>
<p class="note">Keputusan membuang atau menyimpan outlier harus berdasarkan konteks domain, bukan otomatis.</p>
"""
section("c-out", "6 · Outlier", "", "Aturan IQR (dipakai di praktikum) atau z-score.", [
    snip("Aturan IQR", """
        def batas_iqr(series, k=1.5):
            s = series.dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            return q1 - k * iqr, q3 + k * iqr

        def hitung_outlier(series, k=1.5):
            low, high = batas_iqr(series, k)
            s = series.dropna()
            return int(((s < low) | (s > high)).sum())

        for kol in ['age', 'watch_hours']:
            low, high = batas_iqr(df[kol])
            print(f'{kol}: [{low:.2f}, {high:.2f}] -> outlier = {hitung_outlier(df[kol])}')
    """, fig="c_iqr"),
    snip("z-score", """
        z = (df['bmi'] - df['bmi'].mean()) / df['bmi'].std()
        df[z.abs() > 3]                            # |z| > 3 dianggap outlier
    """),
    snip("Tangani outlier", """
        low, high = batas_iqr(df['age'])
        # a) buang (NaN tetap disimpan untuk diimputasi nanti)
        df = df[df['age'].isna() | df['age'].between(low, high)].copy()
        # b) capping / winsorizing
        df['charges'] = df['charges'].clip(lower=low, upper=high)
        # c) jadikan NaN lalu imputasi
        df.loc[~df['age'].between(low, high), 'age'] = np.nan
    """),
    snip("Transformasi log (data miring)", """
        print('skew:', df['watch_hours'].skew())   # > 1 sangat miring, 0.5–1 sedang
        df['watch_log'] = np.log1p(df['watch_hours'])   # log(1 + x), aman untuk 0
    """, fig="c_log"),
    snip("Cek visual", """
        num = ['age', 'bmi', 'charges']
        df[num].plot(kind='box', subplots=True, layout=(1, 3), figsize=(12, 3))
        plt.tight_layout(); plt.show()
    """),
    rich("Buang atau simpan?", OUT_TBL),
])

section("c-kolom", "7 · Buang kolom tidak berguna", "", "", [
    snip("Konstan, ID, redundan", """
        konstan = df.columns[df.nunique() <= 1]                 # variansi 0
        df = df.drop(columns=konstan)
        df = df.drop(columns=['customer_id'])                   # ID unik, bukan fitur
        corr = df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        redundan = [c for c in upper.columns if (upper[c] > 0.9).any()]
        df = df.drop(columns=redundan)
    """),
])

section("c-enc", "8 · Encoding & scaling", "", "Ubah kategori jadi angka dan samakan skala numerik.", [
    snip("Encoding", """
        # ordinal (ada urutan) -> map
        df['plan_enc'] = df['plan'].map({'Basic': 0, 'Standard': 1, 'Premium': 2})
        # nominal (tanpa urutan) -> one-hot
        df = pd.get_dummies(df, columns=['city'], prefix='city', dtype=int)
    """, "Label encoding pada kategori nominal membuat model mengira ada urutan (seolah Medan &gt; Bandung)."),
    snip("Scaling manual", """
        num = ['age', 'watch_hours', 'monthly_spend']
        minmax = (df[num] - df[num].min()) / (df[num].max() - df[num].min())   # [0, 1]
        zscore = (df[num] - df[num].mean()) / df[num].std()                    # mean 0, std 1
    """, fig="c_scale"),
    snip("Scaling sklearn", """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        df[num] = MinMaxScaler().fit_transform(df[num])
        # atau: df[num] = StandardScaler().fit_transform(df[num])
    """, "<code>StandardScaler</code> memakai std populasi (<code>ddof=0</code>); <code>pandas.std()</code> default <code>ddof=1</code>. Hasil sedikit berbeda."),
    snip("Skewness", """
        df['watch_hours'].skew()        # > 0 miring kanan (mean > median), < 0 miring kiri
        df['watch_hours'].plot(kind='hist', bins=30)
        plt.axvline(df['watch_hours'].mean(), color='red', ls='--', label='mean')
        plt.axvline(df['watch_hours'].median(), color='green', ls='--', label='median')
        plt.legend(); plt.show()
    """, fig="c_skew"),
])

section("c-valid", "9 · Validasi & laporan", "", "Buktikan data sudah bersih.", [
    snip("Cek akhir", """
        assert df.isna().sum().sum() == 0, 'masih ada NaN'
        assert not df.duplicated().any(), 'masih ada duplikat'
        assert df['age'].between(0, 120).all()
        df.info(); df.describe()
    """),
    snip("Laporan sebelum → sesudah", """
        n0 = len(df_raw)
        print(f'Baris  : {n0} -> {len(df)} ({n0 - len(df)} dibuang)')
        print(f'Kolom  : {df_raw.shape[1]} -> {df.shape[1]}')
        print('NaN    :', df_raw.isna().sum().sum(), '->', df.isna().sum().sum())
    """),
    snip("Pipeline lengkap (ringkas)", """
        df = df_raw.copy()
        df.columns = df.columns.str.strip().str.lower()
        df['city'] = df['city'].str.strip().str.title()
        df = df.replace(['?', '-', ''], np.nan)
        df['monthly_spend'] = pd.to_numeric(df['monthly_spend'], errors='coerce')
        df = df.drop_duplicates().reset_index(drop=True)
        low, high = batas_iqr(df['age'])
        df = df[df['age'].isna() | df['age'].between(low, high)].copy()
        num = df.select_dtypes('number').columns
        df[num] = df[num].fillna(df[num].median())
        df['city'] = df['city'].fillna(df['city'].mode()[0])
        df = df.drop(columns=df.columns[df.nunique() <= 1])
    """),
])

CERR = """
<div class="tbl"><table>
<thead><tr><th>Gejala</th><th>Penyebab</th><th>Perbaikan</th></tr></thead>
<tbody>
<tr><td><code>isna().sum()</code> = 0 padahal ada <code>"?"</code></td><td>placeholder masih string</td><td><code>replace('?', np.nan)</code> / <code>to_numeric(errors='coerce')</code></td></tr>
<tr><td>Kolom angka ber-dtype <code>object</code></td><td>ada teks di dalamnya</td><td>cari dengan <code>unique()</code>, lalu <code>to_numeric</code></td></tr>
<tr><td><code>fillna(inplace=True)</code> tidak berefek</td><td>copy-on-write (pandas 3)</td><td><code>df['a'] = df['a'].fillna(...)</code></td></tr>
<tr><td><code>SettingWithCopyWarning</code></td><td>mengubah hasil filter</td><td>tambah <code>.copy()</code> setelah filter, atau pakai <code>df.loc[mask, 'a'] = ...</code></td></tr>
<tr><td><code>IntCastingNaNError</code> saat <code>astype(int)</code></td><td>masih ada NaN</td><td>isi dulu, atau <code>astype('Int64')</code></td></tr>
<tr><td><code>fillna(df['c'].mode())</code> hanya mengisi baris pertama</td><td><code>mode()</code> mengembalikan Series</td><td><code>mode()[0]</code></td></tr>
<tr><td><code>AttributeError: .str</code></td><td>kolom bukan string</td><td><code>.astype(str)</code> dulu atau hanya kolom object</td></tr>
<tr><td>Hasil <code>get_dummies</code> True/False</td><td>default bool</td><td><code>dtype=int</code></td></tr>
<tr><td>Tanggal 03/04 terbaca 4 Maret</td><td>format bulan/hari</td><td><code>dayfirst=True</code> atau <code>format='%d/%m/%Y'</code></td></tr>
<tr><td>Rata-rata berubah jauh setelah imputasi</td><td>mean terpengaruh outlier</td><td>pakai median, atau tangani outlier dulu</td></tr>
<tr><td>Jumlah baris turun drastis</td><td><code>dropna()</code> tanpa <code>subset</code></td><td>tentukan kolomnya dengan <code>subset=[...]</code></td></tr>
</tbody></table></div>
"""

JELAS = """
<ul class="tpl">
<li><b>Duplikat</b> dibuang karena menggandakan bobot sampel yang sama sehingga statistik bias.</li>
<li><b>Median</b> dipilih (bukan mean) karena <i>[kolom]</i> miring ke kanan / memiliki outlier; median lebih robust terhadap nilai ekstrem.</li>
<li><b>Modus</b> dipakai untuk <i>[kolom kategorikal]</i> karena nilai kategori tidak bisa dirata-rata.</li>
<li>Outlier <i>[kolom]</i> (<i>[nilai]</i>) adalah galat input yang mustahil secara domain, sehingga <b>dibuang</b>. Outlier <i>[kolom lain]</i> mungkin nyata, sehingga <b>dipertahankan</b>.</li>
<li><i>[kolom]</i> dibuang karena <b>variansinya 0</b> (semua nilai sama) sehingga tidak membedakan observasi.</li>
<li><i>[kolom ordinal]</i> memakai <b>label encoding</b> karena ada urutan; <i>[kolom nominal]</i> memakai <b>one-hot</b> agar model tidak mengira ada urutan.</li>
<li><b>Min-Max</b> memetakan ke [0, 1] (butuh batas tetap); <b>z-score</b> menghasilkan mean 0 dan std 1 (lebih tahan outlier moderat).</li>
<li>Hasil: data berkurang dari <i>[n0]</i> menjadi <i>[n1]</i> baris, tanpa missing value dan duplikat.</li>
</ul>
"""
section("c-err", "Error umum & template penjelasan", "", "", [
    rich("Gejala → perbaikan", CERR),
    rich("Template penjelasan keputusan cleaning", JELAS),
])

# ===========================================================================
# Render
FIGMAP = {
    "Kerangka satu grafik": "anatomy", "Subplot (small multiples)": "subplots",
    "Histogram (numerik)": "hist", "KDE": "kde", "Count plot (kategorikal)": "countplot",
    "Boxplot": "boxplot", "Violin / strip": "violin", "Dot plot (catplot)": "dotplot",
    "Bar chart · seaborn": "bar", "Radial / polar bar": "polar", "Sankey": "sankey",
    "Word cloud": "wordcloud", "Pie / donut": "pie", "Stacked bar": "stacked",
    "Waffle (square pie)": "waffle", "Tree map": "treemap", "Line chart": "line",
    "Area & stacked area": "area", "Candlestick": "candle", "Moving average": "rolling",
    "Scatter": "scatter", "Korelasi + heatmap": "heatmap", "Bubble": "bubble",
    "Scatter matrix": "pairplot",
}
used = set()

# Data shape each chart expects: (raw/aggregated, how to prepare)
DATAMAP = {
    "Histogram (numerik)": ("mentah", "1 kolom numerik, 1 baris = 1 observasi. Tidak perlu <code>groupby</code>: fungsi menghitung bin sendiri."),
    "KDE": ("mentah", "1 kolom numerik (+ kolom kategori untuk <code>hue</code>). Tanpa agregasi."),
    "Count plot (kategorikal)": ("mentah", "1 kolom kategori. <code>countplot</code> menghitung jumlah sendiri, jadi jangan <code>value_counts()</code> dulu."),
    "Boxplot": ("mentah", "kolom numerik + (opsional) kolom kategori untuk grup. Butuh semua nilai, bukan rata-rata."),
    "Violin / strip": ("mentah", "sama seperti boxplot: semua nilai per grup."),
    "Dot plot (catplot)": ("mentah", "1 baris = 1 observasi: kolom numerik, kolom kategori, kolom <code>hue</code>."),
    "Bar chart · seaborn": ("agregat", "1 baris per kategori. Siapkan dengan <code>value_counts()</code> (jumlah) atau <code>groupby('kat')['nilai'].sum()</code>. <code>sns.barplot</code> juga menerima data mentah, tetapi otomatis menghitung <b>mean</b>."),
    "Bar chart · plotly": ("agregat", "1 baris per kategori: kolom label + kolom nilai (hasil <code>groupby().size()</code> / <code>value_counts().reset_index()</code>)."),
    "Radial / polar bar": ("agregat", "1 baris per (arah, kategori) + nilai. Dari data mentah: <code>groupby(['direction','strength'])['frequency'].sum()</code>."),
    "Sankey": ("agregat", "daftar link (asal, tujuan, nilai) dari <code>groupby(['asal','tujuan']).size()</code>, lalu nama node diubah ke indeks."),
    "Word cloud": ("teks", "satu string panjang (<code>' '.join(...)</code>) atau dict <code>{kata: frekuensi}</code> (<code>generate_from_frequencies</code>)."),
    "Pie / donut": ("agregat", "1 baris per kategori: <code>value_counts()</code> atau <code>groupby().sum()</code>."),
    "Stacked bar": ("agregat · wide", "tabel 2 kategori: index = kategori utama, kolom = sub-kategori. <code>pd.crosstab</code> / <code>groupby([a, b]).size().unstack()</code>."),
    "Waffle (square pie)": ("agregat", "dict <code>{label: jumlah}</code> dari <code>value_counts().to_dict()</code>."),
    "Tree map": ("mentah / agregat", "1 baris per daun (leaf) + kolom tiap level hierarki + kolom nilai. Plotly menjumlahkan sendiri per <code>path</code>."),
    "Line chart": ("agregat · waktu", "1 nilai per tanggal, terurut. Banyak transaksi per hari → <code>groupby('tanggal').sum()</code> atau <code>resample('D')</code>. Banyak seri: wide (<code>df.plot</code>) atau long (<code>hue</code>)."),
    "Area & stacked area": ("wide · waktu", "kolom tanggal + 1 kolom per seri. Dari format long: <code>pivot_table(index='tanggal', columns='seri', values='nilai')</code>."),
    "Candlestick": ("agregat · waktu", "1 baris per periode: Open, High, Low, Close. Ganti periode: <code>resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last'})</code>."),
    "Moving average": ("waktu", "Series ber-index tanggal yang <b>lengkap</b> tiap hari (<code>asfreq('D')</code>) agar jendela <code>rolling(n)</code> = n hari."),
    "Scatter": ("mentah", "1 baris = 1 observasi, 2 kolom numerik (+ kategori untuk <code>hue</code>). Jangan diagregasi: tiap titik adalah satu data."),
    "Korelasi + heatmap": ("matriks", "<code>df.corr(numeric_only=True)</code>: hasilnya matriks kolom × kolom."),
    "Heatmap pivot": ("agregat · wide", "matriks 2 kategori: <code>pivot_table(index=a, columns=b, values=v, aggfunc='sum')</code>."),
    "Bubble": ("mentah", "1 baris per entitas (mis. negara pada satu tahun): x, y, size, color. Filter dulu dengan <code>query('year == 2007')</code>."),
    "Scatter matrix": ("mentah", "beberapa kolom numerik + 1 kolom kategori untuk warna. Tanpa agregasi."),
}


def datahint(b):
    h = DATAMAP.get(b["title"])
    if not h:
        return ""
    kind, text = h
    cls = "raw" if kind.startswith("mentah") else "agg"
    return f'<p class="datahint"><span class="dk {cls}">data: {kind}</span>{text}</p>'


def figure(b):
    name = b.get("fig") or FIGMAP.get(b["title"])
    if not name:
        return ""
    used.add(name)
    return f'<figure class="figbox" aria-label="Ilustrasi {html.escape(b["title"])}">{FIGS[name]}</figure>'


def render_block(b):
    if b["kind"] == "html":
        wide = " wide" if b.get("wide", True) else ""
        return (f'<article class="card{wide}" data-search="{html.escape(b.get("search", b["title"]).lower())}">'
                f'<h3>{html.escape(b["title"])}</h3>{figure(b)}{b["body"]}</article>')
    note = f'<p class="note">{b["note"]}</p>' if b["note"] else ""
    return (f'<article class="card" data-search="{html.escape((b["title"] + " " + b["code"]).lower())}">'
            f'<div class="card-h"><h3>{html.escape(b["title"])}</h3>'
            f'<button class="copy" type="button" aria-label="Salin kode {html.escape(b["title"])}">Salin</button></div>'
            f'{figure(b)}{datahint(b)}<pre><code class="language-python">{html.escape(b["code"])}</code></pre>{note}</article>')


# ---- tab 3: prediction questions (prepared by tools/make_prediksi.py) ----
PRED = json.load(open(TOOLS / "prediksi.json", encoding="utf-8"))
for s in PRED:
    sections.append((s["id"], s["title"], s["tag"], s["intro"],
                     [{"kind": "html", "title": b["title"], "body": b["body"], "search": b["search"], "fig": None}
                      for b in s["blocks"]], "pred"))

TABS = [("visualisasi", "Visualisasi", "viz"), ("cleaning", "Cleaning Data", "clean"), ("prediksi", "Prediksi Soal", "pred")]
FLOW = {
    "viz": '<div class="flow"><b>Import</b>→<b>Cek data</b>→<b>Ringkas (groupby/crosstab)</b>→<b>Pilih grafik</b>→<b>Plot + judul/label/legend</b>→<b>Interpretasi</b></div>'
           '<div class="legend"><span><i class="sw c-cat"></i>categorical</span><span><i class="sw c-hie"></i>hierarchical</span><span><i class="sw c-tem"></i>temporal</span><span><i class="sw c-rel"></i>relational</span><span><i class="sw c-spa"></i>spatial</span></div>',
    "clean": '<div class="flow"><b>Diagnosis</b>→<b>Seragamkan</b>→<b>Tipe & NaN</b>→<b>Duplikat</b>→<b>Missing</b>→<b>Outlier</b>→<b>Buang kolom</b>→<b>Encoding/scaling</b>→<b>Validasi</b></div>',
    "pred": '<div class="flow"><b>Setup dataset</b>→<b>Soal 1 cleaning</b>→<b>Soal 2–6 visualisasi</b>→<b>Soal 7 debugging</b>→<b>Tugas analisis</b>→<b>Teori</b></div>',
}
tab_btns, navs, panels = [], [], []
for tid, tlabel, key in TABS:
    secs = [s for s in sections if s[5] == key]
    n_cards = sum(len(s[4]) for s in secs)
    tab_btns.append(f'<button type="button" role="tab" class="tab" id="tab-{tid}" data-tab="{tid}" aria-controls="panel-{tid}">'
                    f'{tlabel}<span>{n_cards}</span></button>')
    navs.append(f'<nav class="idx" data-tab="{tid}" aria-label="Bagian {tlabel}">' +
                "".join(f'<a href="#{sid}" class="{tag}">{html.escape(t)}</a>' for sid, t, tag, _, _, _ in secs) + "</nav>")
    body = []
    for sid, t, tag, intro, blocks, _ in secs:
        dot = f'<span class="dot {tag}"></span>' if tag else ""
        body.append(f'<section id="{sid}" class="sec"><header class="sec-h"><h2>{dot}{html.escape(t)}</h2>'
                    f'<p>{intro}</p></header><div class="cards">{"".join(render_block(b) for b in blocks)}</div></section>')
    panels.append(f'<div class="panel" id="panel-{tid}" role="tabpanel" aria-labelledby="tab-{tid}" data-tab="{tid}">'
                  f'{FLOW[key]}{"".join(body)}</div>')
missing_figs = set(FIGS) - used
if missing_figs:
    print("WARNING unused figs:", sorted(missing_figs))

PAGE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Panduan Praktikum IF5100</title>
<meta name="description" content="Panduan praktikum IF5100: visualisasi data (matplotlib, seaborn, plotly) dan cleaning data dengan pandas.">
<meta name="theme-color" content="#1f77b4">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<script type="module" src="/src/main.js"></script>
</head>
<body>
<header class="masthead"><div class="wrap">
  <h1>Panduan Praktikum IF5100</h1>
  <p>Visualisasi data, cleaning data, dan prediksi soal praktikum · Pemrograman untuk Data Analitik, Week 11–12</p>
</div></header>
<div class="top"><div class="wrap bar">
  <div class="tabs" role="tablist" aria-label="Topik">__TABS__</div>
  <div class="search">
    <label for="q">Cari snippet</label>
    <input id="q" type="search" placeholder="Cari di tab ini…" autocomplete="off">
    <kbd>/</kbd><span class="count" id="count"></span>
  </div>
</div>
<div class="wrap">__NAVS__</div>
</div>

<main class="wrap">
  __PANELS__
  <p class="empty" id="empty">Tidak ada snippet yang cocok di tab ini. Coba kata lain, misalnya nama fungsi (<code>histplot</code>, <code>fillna</code>), atau cari di tab lain.</p>
  <footer>Disusun dari materi IF5100-11 (pre-processing) serta 12a Data Visualization &amp; 12b Independent Study (Andy Kirk 2012; Iliinsky &amp; Steele 2011; Spoerri). Ilustrasi digambar dengan matplotlib dari data contoh. Snippet diuji dengan pandas 3.1, seaborn 0.13, matplotlib 3.11, plotly 7.1.</footer>
</main>
</body>
</html>
"""

page = (PAGE.replace("__TABS__", "".join(tab_btns)).replace("__NAVS__", "".join(navs))
        .replace("__PANELS__", "\n".join(panels)))
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(page)

with open(OUT_PY, "w", encoding="utf-8") as f:
    for t, c in all_snippets:
        f.write(f"# ==== {t}\n{c}\n\n")
print("cards:", sum(len(s[4]) for s in sections), "snippets:", len(all_snippets), "bytes:", len(page))
