"""Figure from results/*.csv (no segmentation or training)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nuclei

summary = pd.read_csv("results/summary.csv")
per = pd.read_csv("results/per_image.csv")
conds = list(dict.fromkeys(summary.condition))
groups = nuclei.GROUPS
names = {"fluor_small": "fluorescence,\nsmall nuclei", "fluor_large": "fluorescence,\nlarge nuclei",
         "tissue_colour": "stained\ntissue", "brightfield": "brightfield"}
n_per_group = per.drop_duplicates("id").group.value_counts()

has_ft = os.path.exists("results/finetune.csv")
fig, axes = plt.subplots(1, 3 if has_ft else 2, figsize=(17 if has_ft else 12, 4.6),
                         gridspec_kw={"width_ratios": [3, 2, 2] if has_ft else [3, 2]})

# 1. accuracy by group and method
ax = axes[0]
w = 0.8 / len(conds)
for k, c in enumerate(conds):
    d = summary[summary.condition == c].set_index("group").reindex(groups)
    ax.bar(np.arange(len(groups)) + (k - (len(conds) - 1) / 2) * w, d.dsb, w,
           yerr=[d.dsb - d.dsb_lo, d.dsb_hi - d.dsb], capsize=2, label=c)
ax.set_xticks(range(len(groups)), [f"{names[g]}\n(n={n_per_group.get(g, 0)})" for g in groups])
ax.set_ylabel("Data Science Bowl score (95% interval)")
ax.set_title("Accuracy by image type")
ax.set_ylim(0, 1)
ax.legend(fontsize=7.5)

# 2. does Cellpose confidence track per-image accuracy?
ax = axes[1]
c = "cellpose nuclei, inverted when bright"
d = per[(per.condition == c)].dropna(subset=["confidence"])
for g in groups:
    x = d[d.group == g]
    ax.scatter(x.confidence, x.dsb_score, s=8, alpha=0.6, label=names[g].replace("\n", " "))
ax.set_xlabel("mean Cellpose cell probability in the image")
ax.set_ylabel("per-image score")
ax.set_title("Confidence vs accuracy\n(nuclei model, inverted when bright)")
ax.legend(fontsize=7)

# 3. fine-tuning on stained tissue
if has_ft:
    ax = axes[2]
    ft = pd.read_csv("results/finetune.csv")
    for v, style in [("raw", "o-"), ("inverted", "s-")]:
        x = ft[ft.variant == v]
        if x.empty:
            continue
        draws = x.groupby(["k", "draw"]).dsb_score.mean().reset_index()
        agg = draws.groupby("k").dsb_score.agg(["mean", "min", "max"])
        ax.errorbar(agg.index, agg["mean"], yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                    fmt=style, capsize=3, label=f"fine-tuned on {v} images")
    for v, ls in [("off the shelf", ":"), ("inverted, no training", "--")]:
        x = ft[ft.variant == v]
        if not x.empty:
            ax.axhline(x.dsb_score.mean(), ls=ls, color="grey", label=v)
    ax.set_xscale("log")
    ax.set_xlabel("annotated training images (stained tissue)")
    ax.set_ylabel("score on 58 held-out tissue images")
    ax.set_title("How many examples fix it?")
    ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig("results/figure.png", dpi=150)
print("saved results/figure.png")
