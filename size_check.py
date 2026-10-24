"""Check: do fine-tuned models score lower only because they use one fixed nucleus size?

Fine-tuned Cellpose models segment with a single diameter taken from their training images, while the base
model estimates the size of every image. This scores the base nuclei model on the 58 held-out tissue images
(inverted) three ways: automatic size per image, one fixed size (mean of the training pool), and the true size
per image. Writes results/size_check.csv.
"""
import numpy as np
import pandas as pd
import torch
from skimage import color
from cellpose import models
import nuclei

g = pd.read_csv("data/groups.csv")
ids = sorted(g[g.group == "tissue_colour"].id)
order = np.random.default_rng(0).permutation(ids)            # the same split as finetune.py
test, pool = list(order[:58]), list(order[58:])
truth = {i: np.load(f"data/truth/{i}.npz")["lab"] for i in ids}


def diameter(lab):
    areas = np.bincount(lab.ravel())[1:]
    return float(np.median(2 * np.sqrt(areas[areas > 0] / np.pi)))


true_d = {i: diameter(truth[i]) for i in ids}
fixed = float(np.mean([true_d[i] for i in pool]))
cp = models.Cellpose(gpu=torch.backends.mps.is_available(), model_type="nuclei")
rows = []
for i in test:
    x = (1 - color.rgb2gray(nuclei.load_image(i))).astype(np.float32)
    for name, run in [("automatic size per image", lambda: cp.eval(x, diameter=None, channels=[0, 0])[0]),
                      ("one fixed size", lambda: cp.cp.eval(x, diameter=fixed, channels=[0, 0])[0]),
                      ("true size per image", lambda: cp.cp.eval(x, diameter=true_d[i], channels=[0, 0])[0])]:
        rows.append({"setting": name, "id": i, **nuclei.scores(truth[i], run().astype(np.uint16))})
d = pd.DataFrame(rows)
d.to_csv("results/size_check.csv", index=False)
for name, x in d.groupby("setting", sort=False):
    m, lo, hi = nuclei.bootstrap_ci(x.dsb_score)
    print(f"{name:26s} {m:.3f} [{lo:.3f}, {hi:.3f}]")
print(f"fixed size used: {fixed:.1f} px; true sizes range {min(true_d.values()):.1f} to {max(true_d.values()):.1f} px")
