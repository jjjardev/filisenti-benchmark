#!/usr/bin/env python3
"""Regenerate benchmark_chart.png from benchmark_results.csv.

The run contains two different tasks, so the chart has two panels:

  A. 3-class track: models with a neutral output, scored on all rows of the
     six full scoring sets.
  B. neutral-free track: binary models, scored only on rows whose gold label
     is negative or positive. Neutral gold rows are removed, and the remaining
     n is printed on the axis.

The gadgets_train_probe column is excluded: it is training data for one model,
not a test set. Scores in panel B are not comparable to panel A.

Usage: python3 make_chart.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CSV = "benchmark_results.csv"
OUT = "benchmark_chart.png"

SETS = [
    "tagasenti_test",
    "hilisenti_test",
    "shopee_binary",
    "mteb_shopee_3class",
    "taglish_annotated_3class",
    "lazada_stars_3class",
]
LABELS = {
    "tagasenti_test": "TagaSenti",
    "hilisenti_test": "Hiligaynon",
    "shopee_binary": "Shopee binary",
    "mteb_shopee_3class": "MTEB Shopee",
    "taglish_annotated_3class": "Taglish annotated",
    "lazada_stars_3class": "Lazada stars",
}

# Models with a real neutral output. MENG21 and scaredmeow/roberta are absent:
# both have two output logits, so their three-class rows are harness artefacts
# (see README, "Method notes").
THREE_CLASS = [
    "jjjardev/filisenti",
    "jjjardev/hilisenti-v1-model",
    "jjjardev/tagasenti_model",
    "immortalizing-legacies/immortalizing-legacies-sentiment",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "tabularisai/multilingual-sentiment-analysis",
    "dost-asti/RoBERTa-tl-sentiment-analysis",
    "sdbrgo/roberta-tagalog-sentiment-multiclass-classifier",
    "magixxixx/roberta-shopee-sentiment-gadgets",
]
BINARY = [
    "The-Thesis-Gods/lazada-xlm-roberta-sentiment",
    "scaredmeow/xlnet-tagalog-sentiment-shopee",
]
SHORT = {
    "The-Thesis-Gods/lazada-xlm-roberta-sentiment": "The-Thesis-Gods/lazada (solid)",
    "scaredmeow/xlnet-tagalog-sentiment-shopee": "scaredmeow/xlnet (hatched)",
}


def main() -> None:
    df = pd.read_csv(CSV)
    f1 = df.pivot_table(index="model", columns="dataset", values="macro_f1")
    n = df.pivot_table(index="model", columns="dataset", values="n")

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    colors = dict(zip(SETS, plt.get_cmap("tab10").colors[: len(SETS)]))

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(16.5, 8.8),
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.30},
    )
    fig.subplots_adjust(left=0.285, right=0.985, top=0.815, bottom=0.185)

    # ---- Panel A: 3-class track -------------------------------------------
    a = f1.loc[THREE_CLASS, SETS].copy()
    a["mean6"] = a.mean(axis=1)
    a = a.sort_values("mean6")
    y = np.arange(len(a))
    height = 0.80 / len(SETS)
    for j, s in enumerate(SETS):
        ax_a.barh(
            y + (j - (len(SETS) - 1) / 2) * height,
            a[s],
            height=height,
            color=colors[s],
            label=LABELS[s],
            edgecolor="white",
            linewidth=0.4,
        )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{m}   ({v:.3f})" for m, v in zip(a.index, a["mean6"])])
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("Macro F1, all three classes")
    ax_a.set_title("A. 3-class track: full scoring sets, 9 models", loc="left", pad=10)
    ax_a.grid(axis="x", alpha=0.25)
    ax_a.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_a.spines[spine].set_visible(False)

    # ---- Panel B: neutral-free track --------------------------------------
    b = f1.loc[BINARY, SETS]
    yb = np.arange(len(SETS))
    height_b = 0.36
    for k, model in enumerate(BINARY):
        vals = b.loc[model, SETS].values
        ax_b.barh(
            yb + (k - 0.5) * height_b,
            vals,
            height=height_b,
            color=[colors[s] for s in SETS],
            alpha=1.0 if k == 0 else 0.55,
            hatch=None if k == 0 else "///",
            edgecolor="white" if k == 0 else "#333333",
            linewidth=0.4,
        )
        for yy, v in zip(yb + (k - 0.5) * height_b, vals):
            ax_b.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=8)

    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(
        [f"{LABELS[s]}  (n={int(n.loc[BINARY[0], s]):,})" for s in SETS]
    )
    ax_b.set_xlim(0, 1.05)
    ax_b.set_xlabel("Macro F1, negative vs positive only")
    ax_b.set_title("B. neutral-free track: 2 binary models", loc="left", pad=10)
    ax_b.grid(axis="x", alpha=0.25)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)

    # ---- Shared legends, below both panels --------------------------------
    set_handles = [Patch(facecolor=colors[s], label=LABELS[s]) for s in SETS]
    fig.legend(
        handles=set_handles,
        loc="lower left",
        bbox_to_anchor=(0.285, 0.005),
        ncol=6,
        fontsize=8.5,
        frameon=False,
        title="test set (bar colour)",
        title_fontsize=8.5,
        alignment="left",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#777777", label=SHORT[BINARY[0]]),
            Patch(facecolor="#777777", alpha=0.55, hatch="///", edgecolor="#333333",
                  label=SHORT[BINARY[1]]),
        ],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.005),
        ncol=1,
        fontsize=8.5,
        frameon=False,
        title="panel B model (bar style)",
        title_fontsize=8.5,
        alignment="left",
    )

    fig.suptitle(
        "FiliSenti Benchmark: one grid, two different tasks",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
    )
    fig.text(
        0.01,
        0.935,
        "Macro F1. Panel A = all rows, three classes. Panel B = neutral gold rows removed, negative vs positive only; scores are not comparable.\n"
        "gadgets_train_probe excluded (training data for one model). MENG21 and scaredmeow/roberta excluded (two-output heads; see README).",
        fontsize=9,
        color="#444444",
        ha="left",
        va="top",
    )
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
