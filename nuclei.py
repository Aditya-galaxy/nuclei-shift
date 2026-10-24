"""Shared pieces: loading images and ground truth, image groups, matching and metrics."""
import os
import numpy as np
import pandas as pd
from skimage import io

ROOT = "data/stage1_train"
GROUPS = ["fluor_small", "fluor_large", "tissue_colour", "brightfield"]
AREA_SPLIT = 700        # px; a clear gap in median nucleus size between small and large fluorescence


def image_ids():
    return sorted(os.listdir(ROOT))


def load_image(i):
    """RGB uint8 image (alpha dropped)."""
    img = io.imread(f"{ROOT}/{i}/images/{i}.png")
    return img[..., :3] if img.ndim == 3 else np.stack([img] * 3, -1)


def load_truth(i):
    """Instance label image: 0 = background, 1..n = one nucleus each."""
    files = sorted(os.listdir(f"{ROOT}/{i}/masks"))
    lab = None
    for k, f in enumerate(files, 1):
        m = io.imread(f"{ROOT}/{i}/masks/{f}") > 0
        if lab is None:
            lab = np.zeros(m.shape, np.uint16)
        lab[m] = k
    return lab


def assign_groups(stats):
    """Group each image by appearance (colour, background) and median nucleus size."""
    def g(r):
        if r.colour > 5:
            return "tissue_colour"
        if r.median_intensity > 100:
            return "brightfield"
        return "fluor_small" if r.median_area < AREA_SPLIT else "fluor_large"
    return stats.assign(group=stats.apply(g, axis=1))


def iou_matrix(truth, pred):
    """IoU between every true and every predicted object, from the joint label histogram."""
    nt, npd = int(truth.max()), int(pred.max())
    if nt == 0 or npd == 0:
        return np.zeros((nt, npd))
    joint = np.bincount(truth.ravel().astype(np.int64) * (npd + 1) + pred.ravel(), minlength=(nt + 1) * (npd + 1))
    joint = joint.reshape(nt + 1, npd + 1)
    inter = joint[1:, 1:].astype(float)
    area_t = joint[1:, :].sum(1, keepdims=True)
    area_p = joint[:, 1:].sum(0, keepdims=True)
    return inter / (area_t + area_p - inter)


def scores(truth, pred, thresholds=np.arange(0.5, 1.0, 0.05)):
    """Per image: the Data Science Bowl score (mean over IoU thresholds of TP/(TP+FP+FN)) and F1 at IoU 0.5.
    At IoU > 0.5 matches are unique, so counting pairs above the threshold is exact."""
    iou = iou_matrix(truth, pred)
    nt, npd = iou.shape
    out = {}
    aps = []
    for t in thresholds:
        tp = int((iou > t).sum())
        fp, fn = npd - tp, nt - tp
        aps.append(tp / (tp + fp + fn) if (tp + fp + fn) else 1.0)
        if abs(t - 0.5) < 1e-9:
            out["f1_50"] = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    out["dsb_score"] = float(np.mean(aps))
    out["n_true"], out["n_pred"] = nt, npd
    return out


def bootstrap_ci(values, n=2000, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), *np.percentile(means, [2.5, 97.5]).tolist()
