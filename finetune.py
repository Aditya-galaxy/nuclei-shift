"""How many annotated images does it take to fix Cellpose on stained tissue?

Stained-tissue images are split once: 58 for testing, 50 as a pool to train from. The Cellpose 'nuclei'
model is fine-tuned on k images from the pool, with the same number of training steps for every k, on either
the raw grey images or brightness-inverted ones. Runs (k, random draw) are listed in RUNS: repeated draws
where the result varies most (small k), one draw elsewhere, to keep the total to about 16 training runs. Each fine-tuned model is scored on the same 58 test images. Model files are deleted after scoring.
Writes results/finetune.csv (one row per test image, k, draw and variant).
"""
import os, shutil, time
import numpy as np
import pandas as pd
import torch
from skimage import color
from cellpose import models, train
import nuclei

GROUP, N_TEST = "tissue_colour", 58
RUNS = {"raw": [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (5, 0), (5, 1), (10, 0), (25, 0), (50, 0)],
        "inverted": [(1, 0), (1, 1), (5, 0), (25, 0), (50, 0)]}
STEPS = dict(n_epochs=int(os.environ.get("FT_EPOCHS", 100)), nimg_per_epoch=32, batch_size=8, learning_rate=0.005, weight_decay=1e-5)
gpu = torch.backends.mps.is_available()

g = pd.read_csv("data/groups.csv")
ids = sorted(g[g.group == GROUP].id)
rng = np.random.default_rng(0)
order = rng.permutation(ids)
test_ids, pool = list(order[:N_TEST]), list(order[N_TEST:])
truth = {i: np.load(f"data/truth/{i}.npz")["lab"] for i in ids}


def grey(i, invert):
    x = color.rgb2gray(nuclei.load_image(i)).astype(np.float32)
    return 1 - x if invert else x


out_path = "results/finetune.csv"
done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()
rows = done.to_dict("records")
t0 = time.time()
for invert in [False, True]:
    variant = "inverted" if invert else "raw"
    test_x = [grey(i, invert) for i in test_ids]
    for k, draw in RUNS[variant]:
        if True:
            if not done.empty and ((done.variant == variant) & (done.k == k) & (done.draw == draw)).any():
                continue
            pick = list(np.random.default_rng(100 * k + draw).choice(pool, k, replace=False))
            model = models.CellposeModel(gpu=gpu, model_type="nuclei")
            save = f"results/ft_tmp/{variant}_{k}_{draw}"
            os.makedirs(save, exist_ok=True)              # Cellpose creates only the last folder level
            path, *_ = train.train_seg(model.net, train_data=[grey(i, invert) for i in pick],
                                       train_labels=[truth[i] for i in pick], channels=[0, 0],
                                       save_path=save, model_name="ft", min_train_masks=1, **STEPS)
            tuned = models.CellposeModel(gpu=gpu, pretrained_model=str(path))
            diam = float(tuned.diam_labels)
            for i, x in zip(test_ids, test_x):
                lab = tuned.eval(x, diameter=diam, channels=[0, 0])[0].astype(np.uint16)
                rows.append({"variant": variant, "k": k, "draw": draw, "id": i, **nuclei.scores(truth[i], lab)})
            shutil.rmtree(save, ignore_errors=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            last = pd.DataFrame(rows)
            last = last[(last.variant == variant) & (last.k == k) & (last.draw == draw)]
            print(f"{variant:8s} k={k:2d} draw={draw}  dsb={last.dsb_score.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)

# baselines on the same test images, from the cached off-the-shelf runs (replaced on every run)
rows = [r for r in rows if r["variant"] in ("raw", "inverted")]
for name, method in [("off the shelf", "cp_nuclei"), ("inverted, no training", "cp_nuclei_inv")]:
    for i in test_ids:
        p = f"results/preds/{method}/{i}.npz"
        if os.path.exists(p):
            rows.append({"variant": name, "k": 0, "draw": 0, "id": i, **nuclei.scores(truth[i], np.load(p)["lab"])})
pd.DataFrame(rows).to_csv(out_path, index=False)
print("done")
