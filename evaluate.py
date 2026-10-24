"""Score every method and condition per image, then summarise per appearance group.

Conditions:
  off the shelf  - one setting for every image
  one adaptation - brightness inverted for bright-background images (Cellpose *_inv runs), plain run otherwise
The classical method always applies the inversion, so it has only the adapted condition.
Writes results/per_image.csv, results/summary.csv, results/summary.md, results/confidence.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import nuclei

groups = pd.read_csv("data/groups.csv").set_index("id")
ids = list(groups.index)
truth = {i: np.load(f"data/truth/{i}.npz")["lab"] for i in ids}


def load(method, i):
    p = f"results/preds/{method}/{i}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["lab"], z["conf"]


CONDITIONS = {                                     # name: (method for dark images, method for bright images)
    "classical": ("classical", "classical"),
    "cellpose nuclei, off the shelf": ("cp_nuclei", "cp_nuclei"),
    "cellpose nuclei, inverted when bright": ("cp_nuclei", "cp_nuclei_inv"),
    "cellpose cyto3, off the shelf": ("cp_cyto3", "cp_cyto3"),
    "cellpose cyto3, inverted when bright": ("cp_cyto3", "cp_cyto3_inv"),
}

rows = []
for name, (dark_m, bright_m) in CONDITIONS.items():
    for i in ids:
        bright = bool(groups.loc[i, "bright"])
        got = load(bright_m if bright else dark_m, i)
        if got is None:
            continue
        lab, conf = got
        s = nuclei.scores(truth[i], lab)
        rows.append({"condition": name, "id": i, "group": groups.loc[i, "group"], **s,
                     "confidence": float(conf.mean()) if conf.size else np.nan})
per = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
per.to_csv("results/per_image.csv", index=False)

summary = []
for (name, g), d in per.groupby(["condition", "group"], sort=False):
    m, lo, hi = nuclei.bootstrap_ci(d.dsb_score)
    f, flo, fhi = nuclei.bootstrap_ci(d.f1_50)
    summary.append({"condition": name, "group": g, "n": len(d), "dsb": m, "dsb_lo": lo, "dsb_hi": hi,
                    "f1": f, "f1_lo": flo, "f1_hi": fhi})
summary = pd.DataFrame(summary)
summary.to_csv("results/summary.csv", index=False)

# confidence: does Cellpose's mean cell probability track per-image accuracy, overall and within groups?
conf_rows = []
for name, d in per.groupby("condition", sort=False):
    d = d.dropna(subset=["confidence"])
    if d.empty:
        continue
    within = [spearmanr(x.confidence, x.dsb_score).statistic for _, x in d.groupby("group")
              if len(x) > 10 and x.confidence.nunique() > 1 and x.dsb_score.nunique() > 1]
    conf_rows.append({"condition": name, "spearman_all": spearmanr(d.confidence, d.dsb_score).statistic,
                      "spearman_within_groups": float(np.nanmean(within))})
pd.DataFrame(conf_rows).to_csv("results/confidence.csv", index=False)

order = nuclei.GROUPS
lines = ["| Method | " + " | ".join(f"{g} (n={int(groups.group.eq(g).sum())})" for g in order) + " |",
         "|---" * (len(order) + 1) + "|"]
for name in CONDITIONS:
    cells = []
    for g in order:
        r = summary[(summary.condition == name) & (summary.group == g)]
        cells.append(f"{r.dsb.iloc[0]:.3f} [{r.dsb_lo.iloc[0]:.3f}, {r.dsb_hi.iloc[0]:.3f}]" if len(r) else "not run")
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
lines.append("\nData Science Bowl score: per image, the mean over IoU thresholds 0.5 to 0.95 of TP / (TP + FP + FN); "
             "averaged over images with a 95% bootstrap interval.")
open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(pd.DataFrame(conf_rows).round(3).to_string(index=False))
