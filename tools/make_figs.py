"""Render small illustrative charts as theme-aware inline SVG strings -> tools/figs.json.
Usage: python tools/make_figs.py   (needs: pip install -r tools/requirements.txt)
Sentinel colours are swapped for CSS variables so text/axes follow the page theme."""
import io, json, re, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import seaborn as sns

FG, LINE, MUTED, SURF = "#010203", "#040506", "#070809", "#fafbfc"
TAB = plt.get_cmap("tab10").colors
plt.rcParams.update({
    "svg.fonttype": "none", "font.size": 7.5, "font.family": "sans-serif",
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "axes.facecolor": "none", "figure.facecolor": "none",
    "savefig.facecolor": "none", "axes.grid": False, "grid.color": LINE, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8,
    "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 6.5,
    "axes.prop_cycle": matplotlib.cycler(color=TAB), "patch.edgecolor": SURF,
    "xtick.major.size": 2, "ytick.major.size": 2,
})
rng = np.random.default_rng(3)
W, H = 3.6, 2.1
figs = {}


def save(name, fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r'<metadata>.*?</metadata>', "", s, flags=re.S)
    s = re.sub(r'\s(width|height)="[\d.]+pt"', "", s, count=2)
    for k, v in {FG: "var(--fg)", LINE: "var(--line)", MUTED: "var(--muted)", SURF: "var(--surface)"}.items():
        s = s.replace(k, v)
    s = s.replace("<svg ", '<svg role="img" class="fig" ', 1)
    s = re.sub(r">\s+<", "><", s)
    figs[name] = s


def ax1(w=W, h=H, **kw):
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    return fig, ax


def yzero(ax):
    ax.set_ylim(bottom=0)


# ---------------- distribution ----------------
age = np.concatenate([rng.gamma(4, 8, 400), rng.normal(70, 6, 40)])
fig, ax = ax1(); ax.hist(age, bins=20, color=TAB[0], edgecolor=SURF, linewidth=.5)
ax.set_xlabel("age"); ax.set_ylabel("count"); save("hist", fig)

fig, ax = ax1()
for i, (m, s, lab) in enumerate([(28, 6, "no"), (36, 5, "yes")]):
    x = np.linspace(5, 60, 200); y = np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
    ax.fill_between(x, y, alpha=.35, color=TAB[i], label=lab); ax.plot(x, y, color=TAB[i], lw=1)
ax.set_xlabel("bmi"); ax.set_ylabel("density"); ax.legend(title="smoker", title_fontsize=6.5); yzero(ax); save("kde", fig)

fig, ax = ax1(); days = ["Thur", "Fri", "Sat", "Sun"]; v = [62, 19, 87, 76]
ax.bar(days, v, color=TAB[0], width=.65); ax.set_ylabel("count"); ax.set_xlabel("day"); save("countplot", fig)

fig, ax = ax1()
data = [rng.gamma(2, 4000, 200), np.concatenate([rng.normal(32000, 7000, 60), [62000, 64000]])]
bp = ax.boxplot(data, tick_labels=["no", "yes"], patch_artist=True, widths=.5,
                medianprops=dict(color=FG), flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none"),
                whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for p, c in zip(bp["boxes"], TAB): p.set_facecolor(c); p.set_alpha(.75); p.set_edgecolor(MUTED)
ax.set_xlabel("smoker"); ax.set_ylabel("charges"); save("boxplot", fig)

fig, ax = ax1()
vp = ax.violinplot(data, showmedians=True)
for b, c in zip(vp["bodies"], TAB): b.set_facecolor(c); b.set_alpha(.6); b.set_edgecolor("none")
for k in ("cbars", "cmins", "cmaxes", "cmedians"): vp[k].set_color(MUTED); vp[k].set_linewidth(.8)
ax.set_xticks([1, 2], ["no", "yes"]); ax.set_xlabel("smoker"); ax.set_ylabel("charges"); save("violin", fig)

fig, ax = ax1()
provs = ["Jakarta", "Jabar", "Banten", "Jatim", "Bali"]
for i, p in enumerate(provs):
    n = [40, 28, 20, 16, 10][i]
    for j, st in enumerate(["isolated", "released", "deceased"]):
        k = max(1, int(n * [0.6, 0.25, 0.15][j]))
        ax.scatter(rng.integers(15, 85, k), i + rng.uniform(-.18, .18, k), s=9, color=TAB[j], label=st if i == 0 else None)
ax.set_yticks(range(5), provs); ax.set_xlabel("age"); ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=3, fontsize=6, title="current_state", title_fontsize=6); ax.set_ylim(-.6, 4.6); save("dotplot", fig)

# ---------------- categorical ----------------
fig, ax = ax1(); cnt = pd.Series([152, 98, 61, 47, 30, 18], index=["Jakarta", "Jabar", "Banten", "Jatim", "Jateng", "Bali"])
b = ax.barh(cnt.index[::-1], cnt.values[::-1], color=TAB[0], height=.65); ax.bar_label(b, fontsize=6.5, padding=2, color=MUTED)
ax.set_xlabel("count"); save("bar", fig)

fig = plt.figure(figsize=(2.4, 2.4)); ax = fig.add_subplot(projection="polar")
theta = np.deg2rad(np.arange(0, 360, 45)); dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
bottom = np.zeros(8); pal = plt.get_cmap("plasma_r")
for i in range(3):
    r = rng.uniform(.5, 2.2, 8) * (3 - i)
    ax.bar(theta, r, width=.7, bottom=bottom, color=pal(.2 + .3 * i), edgecolor=SURF, linewidth=.4); bottom += r
ax.set_xticks(theta, dirs); ax.set_yticklabels([]); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.grid(color=LINE, lw=.5); ax.spines["polar"].set_color(LINE); save("polar", fig)


def sankey_ax(ax, nodes, links):
    """nodes: {name: (x, y0, y1, color)}, links: [(src, tgt, s0, s1, t0, t1)]"""
    for src, tgt, s0, s1, t0, t1 in links:
        x0 = nodes[src][0] + .06; x1 = nodes[tgt][0]; xm = (x0 + x1) / 2
        verts = [(x0, s0), (xm, s0), (xm, t0), (x1, t0), (x1, t1), (xm, t1), (xm, s1), (x0, s1), (x0, s0)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc=nodes[src][3], alpha=.35, ec="none"))
    for name, (x, y0, y1, c) in nodes.items():
        ax.add_patch(Rectangle((x, y0), .06, y1 - y0, fc=c, ec="none"))
        ha, dx = ("right", -.02) if x > .5 else ("left", .08)
        ax.text(x + dx, (y0 + y1) / 2, name, va="center", ha=ha, fontsize=6.5)


fig, ax = ax1()
nodes = {"Coal": (0, .55, 1, TAB[7]), "Gas": (0, .25, .5, TAB[1]), "Solar": (0, 0, .2, TAB[8]),
         "Electricity": (.47, .2, .85, TAB[0]), "Heat": (.47, 0, .15, TAB[3]),
         "Homes": (.94, .45, 1, TAB[2]), "Industry": (.94, 0, .4, TAB[4])}
links = [("Coal", "Electricity", .6, 1, .45, .85), ("Coal", "Heat", .55, .6, .1, .15), ("Gas", "Electricity", .3, .5, .25, .45),
         ("Gas", "Heat", .25, .3, 0, .05), ("Solar", "Electricity", 0, .2, .2, .25), ("Electricity", "Homes", .5, .85, .55, .9),
         ("Electricity", "Industry", .2, .5, .05, .35), ("Heat", "Homes", .07, .15, .45, .53), ("Heat", "Industry", 0, .07, .35, .4)]
# re-scale manually so links read cleanly
links = [("Coal", "Electricity", .6, 1, .45, .85), ("Coal", "Heat", .55, .6, .08, .13), ("Gas", "Electricity", .30, .5, .25, .45),
         ("Gas", "Heat", .25, .30, .03, .08), ("Solar", "Electricity", .02, .18, .2, .25),
         ("Electricity", "Homes", .5, .85, .6, .95), ("Electricity", "Industry", .2, .5, .05, .35),
         ("Heat", "Homes", .07, .13, .5, .56), ("Heat", "Industry", .01, .07, .35, .39)]
nodes["Homes"] = (.94, .5, .95, TAB[2]); nodes["Industry"] = (.94, .05, .39, TAB[4]); nodes["Heat"] = (.47, .01, .13, TAB[3])
sankey_ax(ax, nodes, links); ax.set_xlim(-.02, 1.02); ax.set_ylim(-.03, 1.03); ax.axis("off"); save("sankey", fig)

from wordcloud import WordCloud
words = {"smart": 30, "watch": 22, "board": 20, "game": 26, "camera": 15, "bike": 12, "travel": 14, "light": 11,
         "wallet": 9, "card": 17, "pen": 8, "speaker": 10, "bag": 9, "kit": 13, "design": 7, "film": 12, "book": 16, "music": 8}
wc = WordCloud(width=520, height=260, mode="RGBA", background_color=None, colormap="tab10",
               prefer_horizontal=.8, random_state=2).generate_from_frequencies(words)
fig, ax = ax1(); ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); save("wordcloud", fig)

# ---------------- hierarchical ----------------
fig, ax = ax1(2.4, 2.1); vals = [45, 25, 15, 10, 5]; labs = ["Jakarta", "Jabar", "Banten", "Jatim", "lain"]
w, t, at = ax.pie(vals, labels=labs, autopct="%1.0f%%", startangle=90, counterclock=False,
                  wedgeprops=dict(width=.45, edgecolor=SURF, linewidth=1), pctdistance=.78, textprops=dict(fontsize=6.5))
for a in at: a.set_color("white"); a.set_fontsize(6)
save("pie", fig)

fig, ax = ax1(); f = np.array([60, 41, 25, 22, 12]); m = np.array([70, 50, 28, 20, 15])
ax.bar(labs, f, color=TAB[0], width=.6, label="female"); ax.bar(labs, m, bottom=f, color=TAB[1], width=.6, label="male")
ax.legend(title="gender", title_fontsize=6.5); ax.set_ylabel("count"); save("stacked", fig)

fig, ax = ax1(W, 1.5); counts = {"Petrol": 50, "Diesel": 14, "CNG": 6}; cols = 14; i = 0
for k, (lab, n) in enumerate(counts.items()):
    for _ in range(n):
        r, c = divmod(i, cols); ax.add_patch(Rectangle((c, -r), .86, .86, color=TAB[k])); i += 1
    ax.add_patch(Rectangle((cols + .8, -k * 1.4), .8, .8, color=TAB[k])); ax.text(cols + 1.9, -k * 1.4 + .4, f"{lab} ({n})", va="center", fontsize=6.5)
ax.set_xlim(-.2, cols + 6.5); ax.set_ylim(-4.3, 1); ax.set_aspect("equal"); ax.axis("off"); save("waffle", fig)

fig, ax = ax1(); cont = {"Asia": [1300, 1100, 230, 190, 160], "Americas": [300, 190, 110, 45], "Africa": [135, 80, 75, 50], "Europe": [82, 65, 60, 59]}
total = sum(map(sum, cont.values())); x0 = 0; cmap = plt.get_cmap("RdBu")
for ci, (cn, cs) in enumerate(cont.items()):
    wdt = sum(cs) / total; y0 = 0
    for j, v in enumerate(cs):
        hgt = v / sum(cs); u = rng.uniform(.06, .3); colr = cmap(u if rng.random() < .5 else 1 - u)
        ax.add_patch(Rectangle((x0, y0), wdt, hgt, fc=colr, ec=SURF, lw=1.2)); y0 += hgt
    ax.text(x0 + .012, .975, cn, va="top", fontsize=6.5, weight="bold", color="black", bbox=dict(fc="white", ec="none", alpha=.8, pad=1.2))
    x0 += wdt
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); save("treemap", fig)

# ---------------- temporal ----------------
d = pd.date_range("2020-03-02", periods=30); conf = np.cumsum(rng.integers(5, 60, 30)) * np.linspace(1, 3, 30)
neg = np.cumsum(rng.integers(40, 140, 30)) * 1.2; tested = conf + neg; dead = conf * .09
fig, ax = ax1()
for y, lab in [(tested, "tested"), (neg, "negative"), (conf, "confirmed"), (dead, "deceased")]: ax.plot(d, y, lw=1.3, label=lab)
ax.legend(); ax.set_ylabel("case"); ax.tick_params(axis="x", rotation=30); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b")); save("line", fig)

fig, ax = ax1()
ax.stackplot(d, dead, conf, neg - conf.clip(max=neg) * 0 , labels=["deceased", "confirmed", "negative"], colors=[TAB[3], TAB[1], TAB[2]], alpha=.75)
ax.legend(loc="upper left"); ax.set_ylabel("case"); yzero(ax); ax.tick_params(axis="x", rotation=30); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b")); save("area", fig)

fig, ax = ax1(); n = 28; o = 100 + np.cumsum(rng.normal(0, 1.4, n)); c = o + rng.normal(0, 1.6, n)
hi = np.maximum(o, c) + rng.uniform(.2, 1.4, n); lo = np.minimum(o, c) - rng.uniform(.2, 1.4, n)
for i in range(n):
    col = TAB[2] if c[i] >= o[i] else TAB[3]
    ax.vlines(i, lo[i], hi[i], color=col, lw=.8); ax.add_patch(Rectangle((i - .32, min(o[i], c[i])), .64, max(abs(c[i] - o[i]), .08), color=col))
ax.set_xlim(-1, n); ax.set_ylim(lo.min() - 1, hi.max() + 1); ax.set_ylabel("price"); ax.set_xticks([]); ax.set_xlabel("date"); save("candle", fig)

fig, ax = ax1(); s = pd.Series(100 + np.cumsum(rng.normal(.05, 1.5, 300)))
ax.plot(s.values, color=TAB[0], alpha=.35, lw=.8, label="harian"); ax.plot(s.rolling(30).mean().values, color=TAB[1], lw=1.6, label="MA-30")
ax.legend(); ax.set_ylabel("Close"); ax.set_xticks([]); ax.set_xlabel("date"); save("rolling", fig)

fig, axs = plt.subplots(2, 2, figsize=(W, H), sharex=True)
for a, y, lab, col in zip(axs.flat, [dead, conf, neg, tested], ["Deceased", "Confirmed", "Negative", "Tested"], [TAB[3], TAB[1], TAB[2], TAB[0]]):
    a.fill_between(range(30), y, color=col, alpha=.6, lw=0); a.set_title(lab, fontsize=7); a.set_xticks([]); a.tick_params(labelsize=5.5); a.set_ylim(bottom=0)
fig.tight_layout(pad=.4); save("subplots", fig)

# ---------------- relational ----------------
bmi = rng.normal(30, 6, 220); smoke = rng.random(220) < .22
charges = np.where(smoke, 12000 + bmi * 900, 2000 + bmi * 220) + rng.normal(0, 3000, 220)
fig, ax = ax1()
for k, (msk, lab) in enumerate([(~smoke, "no"), (smoke, "yes")]): ax.scatter(bmi[msk], charges[msk], s=7, alpha=.75, color=TAB[k], label=lab, lw=0)
ax.legend(title="smoker", title_fontsize=6.5); ax.set_xlabel("bmi"); ax.set_ylabel("charges"); save("scatter", fig)

fig, ax = ax1(); x = rng.normal(30, 6, 150); y = 3000 + 380 * x + rng.normal(0, 3500, 150)
ax.scatter(x, y, s=7, alpha=.5, color=TAB[0], lw=0); bb = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
ax.fill_between(xs, np.polyval(bb, xs) - 1200, np.polyval(bb, xs) + 1200, color=TAB[0], alpha=.15, lw=0); ax.plot(xs, np.polyval(bb, xs), color=TAB[0], lw=1.4)
ax.set_xlabel("bmi"); ax.set_ylabel("charges"); save("regplot", fig)

cols = ["age", "bmi", "children", "charges"]; C = np.array([[1, .11, .04, .30], [.11, 1, .01, .20], [.04, .01, 1, .07], [.30, .20, .07, 1]])
fig, ax = ax1(2.9, 2.3); im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
for i in range(4):
    for j in range(4): ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="black" if abs(C[i, j]) < .7 else "white")
ax.set_xticks(range(4), cols, rotation=30); ax.set_yticks(range(4), cols); ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=0); save("heatmap", fig)

fig, ax = ax1(); conts = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
for k, cn in enumerate(conts):
    n = [12, 9, 12, 8, 2][k]; g = 10 ** rng.uniform([2.8, 4.2, 2.6, 3.4, 4.3][k], [4.3, 4.7, 3.8, 4.4, 4.5][k], n)
    le = 30 + 11 * np.log10(g) + rng.normal(0, 3, n); pop = 10 ** rng.uniform(6.5, 9.1, n)
    ax.scatter(g, le, s=pop / 8e6, alpha=.6, color=TAB[k], label=cn, ec=SURF, lw=.4)
ax.set_xscale("log"); ax.set_xlabel("gdpPercap (log)"); ax.set_ylabel("lifeExp"); ax.legend(markerscale=.35, ncol=1, loc="lower right", fontsize=6); save("bubble", fig)

fig, axs = plt.subplots(3, 3, figsize=(2.9, 2.6)); X = rng.normal(0, 1, (120, 3)); X[:, 1] += .7 * X[:, 0]; X[:, 2] += .4 * X[:, 1]
lab = rng.random(120) < .35; X[lab] += .9; names = ["Glucose", "BMI", "Age"]
for i in range(3):
    for j in range(3):
        a = axs[i, j]; a.set_xticks([]); a.set_yticks([])
        if j > i: a.axis("off"); continue
        if i == j:
            a.hist(X[~lab, i], bins=12, color=TAB[0], alpha=.6); a.hist(X[lab, i], bins=12, color=TAB[3], alpha=.6)
        else:
            a.scatter(X[~lab, j], X[~lab, i], s=2, color=TAB[0], lw=0); a.scatter(X[lab, j], X[lab, i], s=2, color=TAB[3], lw=0)
        if j == 0: a.set_ylabel(names[i], fontsize=6)
        if i == 2: a.set_xlabel(names[j], fontsize=6)
fig.tight_layout(pad=.3); save("pairplot", fig)

# ---------------- anatomy ----------------
fig, ax = ax1(W + .3, H + .2)
ax.bar(["A", "B", "C", "D"], [5, 8, 3, 6], color=TAB[0], width=.55, label="2024"); ax.bar(["A", "B", "C", "D"], [3, 2, 4, 2], bottom=[5, 8, 3, 6], color=TAB[1], width=.55, label="2025")
ax.set_title("ax.set_title()"); ax.set_xlabel("ax.set_xlabel()"); ax.set_ylabel("ax.set_ylabel()"); ax.legend(title="ax.legend()", title_fontsize=6.5, loc="upper right")
ax.set_ylim(0, 13); save("anatomy", fig)

# ---------------- cleaning ----------------
fig, ax = ax1(); miss = pd.Series({"monthly_spend": 31, "watch_hours": 18, "age": 12, "city": 4, "plan": 0, "customer_id": 0}).sort_values()
cl = [TAB[3] if v > 25 else (TAB[1] if v > 0 else TAB[7]) for v in miss.values]
b = ax.barh(miss.index, miss.values, color=cl, height=.6); ax.bar_label(b, labels=[f"{v}%" for v in miss.values], fontsize=6.5, padding=2, color=MUTED)
ax.set_xlabel("% missing"); ax.set_xlim(0, 38); save("c_missing", fig)

fig, ax = ax1(W, 1.6); s = np.concatenate([rng.normal(33, 7, 200), [72, 78, 4]])
q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1; lo_, hi_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
ax.boxplot(s, orientation="horizontal", widths=.45, patch_artist=True, boxprops=dict(facecolor=TAB[0], alpha=.35, edgecolor=MUTED),
           medianprops=dict(color=FG), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED), flierprops=dict(marker="o", ms=4, mfc=TAB[3], mec="none"))
for xv, lab in [(lo_, "Q1 − 1.5·IQR"), (q1, "Q1"), (q3, "Q3"), (hi_, "Q3 + 1.5·IQR")]:
    ax.axvline(xv, color=LINE if "IQR" not in lab else TAB[3], lw=.8, ls="--" if "IQR" in lab else ":")
    ax.text(xv, 1.42, lab, ha="center", fontsize=6, color=TAB[3] if "IQR" in lab else MUTED)
ax.text(q1 + iqr / 2, .62, "IQR", ha="center", fontsize=6.5, color=FG); ax.annotate("", xy=(q1, .7), xytext=(q3, .7), arrowprops=dict(arrowstyle="<->", color=FG, lw=.6))
ax.text(76, .82, "outlier", ha="center", fontsize=6.5, color=TAB[3])
ax.set_yticks([]); ax.set_ylim(.5, 1.6); ax.spines["left"].set_visible(False); ax.set_xlabel("age"); save("c_iqr", fig)

fig, ax = ax1(); w_ = rng.gamma(2, 18, 600)
ax.hist(w_, bins=30, color=TAB[0], alpha=.7, edgecolor=SURF, linewidth=.4)
ax.axvline(np.mean(w_), color=TAB[3], lw=1.3, ls="--", label=f"mean {np.mean(w_):.0f}"); ax.axvline(np.median(w_), color=TAB[2], lw=1.3, ls="--", label=f"median {np.median(w_):.0f}")
ax.legend(); ax.set_xlabel("watch_hours (skew > 0)"); ax.set_ylabel("count"); save("c_skew", fig)

fig, axs = plt.subplots(1, 3, figsize=(W + .4, 1.5), sharey=True); raw = rng.gamma(3, 15, 400)
for a, (vals_, t_) in zip(axs, [(raw, "asli"), ((raw - raw.min()) / (raw.max() - raw.min()), "Min-Max [0,1]"), ((raw - raw.mean()) / raw.std(), "z-score (μ=0, σ=1)")]):
    a.hist(vals_, bins=20, color=TAB[0], alpha=.75, edgecolor=SURF, linewidth=.3); a.set_title(t_, fontsize=6.5); a.tick_params(labelsize=5.5); a.set_yticks([])
fig.tight_layout(pad=.3); save("c_scale", fig)

fig, axs = plt.subplots(1, 2, figsize=(W, 1.5), sharey=True)
axs[0].hist(w_, bins=25, color=TAB[0], alpha=.75, edgecolor=SURF, linewidth=.3); axs[0].set_title("asli (miring kanan)", fontsize=6.5)
axs[1].hist(np.log1p(w_), bins=25, color=TAB[2], alpha=.75, edgecolor=SURF, linewidth=.3); axs[1].set_title("np.log1p (lebih simetris)", fontsize=6.5)
for a in axs: a.set_yticks([]); a.tick_params(labelsize=5.5)
fig.tight_layout(pad=.3); save("c_log", fig)

fig, ax = ax1(W, 1.7); t = np.arange(20); y = 10 + np.sin(t / 3) * 4 + t * .3; yn = y.copy(); gap_ = [5, 6, 7, 13]; yn[gap_] = np.nan
ser = pd.Series(yn); ax.plot(t, ser.interpolate(), color=TAB[2], lw=1.2, ls="--", label="interpolate()")
ax.plot(t, ser.ffill(), color=TAB[1], lw=1, drawstyle="steps-post", alpha=.8, label="ffill()")
ax.plot(t, yn, color=TAB[0], lw=1.6, marker="o", ms=3, label="data (ada NaN)"); ax.legend(ncol=3, loc="upper left", fontsize=6); ax.set_ylim(4, 22); ax.set_xlabel("waktu"); save("c_ts", fig)

out = sys.argv[1] if len(sys.argv) > 1 else __import__("pathlib").Path(__file__).with_name("figs.json")
json.dump(figs, open(out, "w", encoding="utf-8"))
print(len(figs), "figs,", sum(map(len, figs.values())) // 1024, "KB total")
for k, v in figs.items():
    print(f"  {k:10s} {len(v)//1024:4d} KB")
