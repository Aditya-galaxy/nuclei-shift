"""Run each segmentation method on every image and cache the predicted label images.

Methods, all used off the shelf with one setting for every image (that is the point of the test):
  classical      - Otsu threshold + distance-transform watershed (scikit-image)
  cp_nuclei      - Cellpose 3 'nuclei' model, diameter estimated automatically
  cp_cyto3       - Cellpose 3 'cyto3' generalist model, diameter estimated automatically
  cp_nuclei_inv, cp_cyto3_inv
                 - the same models with brightness inverted, run only on images with a bright background
                   (grey median > 0.4, the same rule the classical method uses). Combined with the plain runs on
                   dark images, they give the "one adaptation" condition.
Cellpose also saves, per predicted nucleus, its mean cell probability: used later as a confidence score.
Usage: python segment.py classical|cp_nuclei|cp_cyto3|cp_nuclei_inv|cp_cyto3_inv
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, filters, morphology, segmentation, feature
import nuclei

method = sys.argv[1]
out_dir = f"results/preds/{method}"
os.makedirs(out_dir, exist_ok=True)
ids = nuclei.image_ids()


def to_gray(img):
    return color.rgb2gray(img)


def bright_background(img):
    return np.median(to_gray(img)) > 0.4


def classical(img):
    g = to_gray(img)
    if bright_background(img):                   # bright background: nuclei are dark
        g = 1 - g
    mask = g > filters.threshold_otsu(g)
    mask = morphology.remove_small_objects(mask, 20)
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=5, labels=ndi.label(mask)[0], exclude_border=False)
    markers = np.zeros(mask.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask).astype(np.uint16), None


if method.startswith("cp_"):
    from cellpose import models
    import torch
    invert = method.endswith("_inv")
    model = models.Cellpose(gpu=torch.backends.mps.is_available(), model_type=method[3:].replace("_inv", ""))

    def run(img):
        masks, flows, _, _ = model.eval(to_gray(img).astype(np.float32), diameter=None, channels=[0, 0], invert=invert)
        prob = 1 / (1 + np.exp(-flows[2]))          # cell probability per pixel
        conf = ndi.mean(prob, labels=masks, index=np.arange(1, masks.max() + 1)) if masks.max() else np.array([])
        return masks.astype(np.uint16), np.asarray(conf, np.float32)
else:
    run = classical

t0 = time.time()
for k, i in enumerate(ids):
    path = f"{out_dir}/{i}.npz"
    if os.path.exists(path):
        continue
    img = nuclei.load_image(i)
    if method.endswith("_inv") and not bright_background(img):
        continue                                  # dark images use the plain run
    lab, conf = run(img)
    np.savez_compressed(path, lab=lab, conf=conf if conf is not None else np.array([]))
    if (k + 1) % 100 == 0:
        print(f"{method}: {k + 1}/{len(ids)} images, {time.time() - t0:.0f}s", flush=True)
print(f"{method}: done in {time.time() - t0:.0f}s")
