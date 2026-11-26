# -*- coding: utf-8 -*-
"""Gaya grafik seragam untuk seluruh notebook (aturan sumbu dan pelabelan).
- Bar/column: sumbu nilai selalu mulai dari 0. Line chart boleh tidak mulai dari 0 bila rentangnya wajar.
- Interval skala seragam (kelipatan 1/2/2,5/5 x 10^n), angka ringkas (rb, Jt, M), tanpa sumbu ganda.
- Judul memuat metrik, kategori, dan periode; label sumbu memuat satuan; semua teks mendatar.
- Legenda hanya bila seri > 1; sumber data dicantumkan di kaki grafik."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SUMBER = "Sumber: dataset penjualan studi kasus Bengkel Sumber Urip (olah data penulis), data hingga 31 Okt 2026"
PAL = ["#1F4E78", "#E8710A", "#2F855A", "#7C3AED"]

def rcparams():
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 160, "font.size": 9.5, "axes.titlesize": 11.5, "axes.labelsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                         "axes.axisbelow": True})

def _n(x):
    return f"{x:.1f}".rstrip("0").rstrip(".").replace(".", ",")

def singkat(v, _=None):
    a = abs(v)
    if a >= 1e9: return _n(v / 1e9) + " M"
    if a >= 1e6: return _n(v / 1e6) + " Jt"
    if a >= 1e3: return _n(v / 1e3) + " rb"
    return _n(v)

def ribuan(v, _=None):
    return f"{v:,.0f}".replace(",", ".")

def tgl(d):
    return f"{d.day} {BULAN[d.month - 1]}"

def langkah(rentang, target=5):
    raw = max(rentang, 1e-9) / target
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw: return m * mag

def _akhir(fig, ax, judul, sumber=SUMBER):
    ax.set_title(judul, loc="left", fontweight="bold")
    fig.text(0.01, 0.012, sumber, fontsize=7.5, color="#6B7280", ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

def hbar(labels, values, judul, xlabel, path, fmt=singkat, label_fmt=None, warna=PAL[0]):
    """Batang horizontal (label kategori panjang tetap terbaca mendatar). Sumbu nilai mulai 0."""
    order = np.argsort(values)
    labels, values = [labels[i] for i in order], [values[i] for i in order]
    label_fmt = label_fmt or fmt
    fig, ax = plt.subplots(figsize=(8, 0.46 * len(labels) + 2.0))
    st = langkah(max(values), 8); top = np.ceil(max(values) * 1.14 / st) * st
    ax.barh(labels, values, color=warna, height=.62)
    ax.set_xlim(0, top); ax.xaxis.set_major_locator(MultipleLocator(st)); ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.grid(axis="y", visible=False); ax.set_xlabel(xlabel); ax.set_ylabel("")
    for i, v in enumerate(values): ax.text(v + top * .012, i, label_fmt(v), va="center", fontsize=8.5)
    _akhir(fig, ax, judul); fig.savefig(path); plt.show()

def vbar(labels, values, judul, ylabel, path, fmt=singkat, label_fmt=None, warna=PAL[1]):
    """Kolom vertikal untuk kategori pendek. Sumbu Y mulai 0."""
    label_fmt = label_fmt or fmt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    st = langkah(max(values)); top = np.ceil(max(values) * 1.14 / st) * st
    ax.bar(labels, values, color=warna, width=.55)
    ax.set_ylim(0, top); ax.yaxis.set_major_locator(MultipleLocator(st)); ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.grid(axis="x", visible=False); ax.set_ylabel(ylabel); ax.set_xlabel("")
    for i, v in enumerate(values): ax.text(i, v + top * .015, label_fmt(v), ha="center", fontsize=8.5)
    _akhir(fig, ax, judul); fig.savefig(path); plt.show()

def garis(series, judul, ylabel, path, fmt=singkat, ymin=None, putus=(), titik_tanggal=True):
    """series: {nama: (x_tanggal, y)}. Y boleh tidak dari 0 (rentang wajar, interval seragam). Legenda hanya bila seri > 1."""
    allv = np.concatenate([np.asarray(y, float) for _, y in series.values()])
    lo, hi = float(allv.min()), float(allv.max())
    st = langkah(hi - lo if ymin is None else hi - ymin)
    y0 = ymin if ymin is not None else np.floor((lo - .12 * (hi - lo)) / st) * st
    y1 = np.ceil((hi + .12 * (hi - lo)) / st) * st
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (nm, (x, y)) in enumerate(series.items()):
        ax.plot(x, y, marker="o", ms=4.5, lw=2, color=PAL[k % len(PAL)], ls="--" if nm in putus else "-", label=nm)
    ax.set_ylim(max(y0, 0) if lo >= 0 else y0, y1); ax.yaxis.set_major_locator(MultipleLocator(st)); ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    xs = sorted(set(np.concatenate([np.asarray(x) for x, _ in series.values()])))
    ax.set_xticks(xs); ax.set_xticklabels([tgl(__import__("pandas").Timestamp(d)) for d in xs], fontsize=8)
    ax.grid(axis="x", visible=False); ax.set_ylabel(ylabel); ax.set_xlabel("Awal minggu (tanggal, 2026)")
    if len(series) > 1: ax.legend(frameon=False, ncol=len(series), loc="upper left", bbox_to_anchor=(0, 1.0))
    _akhir(fig, ax, judul); fig.savefig(path); plt.show()
