"""Batch nucleus segmentation for a folder of images: the deployable part of the project.

  python predict.py --input images/ --output out/ [--model nuclei|cyto3] [--invert auto|never|always]
                    [--device cpu|mps|cuda]

For every image it writes <name>_labels.png (16-bit instance labels) and adds a row to out/summary.csv:
number of nuclei, median nucleus area, mean Cellpose confidence, whether it was inverted, seconds taken.
--invert auto inverts brightness when the image background is bright (grey median > 0.4), the rule that
recovered Cellpose on stained tissue in this project's benchmark.

Sharding: when run as one of several parallel tasks (a Cloud Run Job sets CLOUD_RUN_TASK_INDEX and
CLOUD_RUN_TASK_COUNT), each task takes every N-th image and writes summary-<index>.csv.
"""
import argparse, csv, os, time
import numpy as np
from scipy import ndimage as ndi
from skimage import color, io


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="nuclei", choices=["nuclei", "cyto3"])
    ap.add_argument("--invert", default="auto", choices=["auto", "never", "always"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    a = ap.parse_args()

    import torch
    from cellpose import models
    model = models.Cellpose(gpu=a.device != "cpu", device=torch.device(a.device), model_type=a.model)
    os.makedirs(a.output, exist_ok=True)
    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg")))
    task, n_tasks = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0)), int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))
    files = files[task::n_tasks]
    name = "summary.csv" if n_tasks == 1 else f"summary-{task}.csv"
    started = time.perf_counter()
    with open(os.path.join(a.output, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "n_nuclei", "median_area_px", "mean_confidence", "inverted", "seconds"])
        for f in files:
            t = time.perf_counter()
            img = io.imread(os.path.join(a.input, f))
            grey = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(np.float32) / max(img.max(), 1)
            inv = a.invert == "always" or (a.invert == "auto" and np.median(grey) > 0.4)
            masks, flows, _, _ = model.eval(grey.astype(np.float32), diameter=None, channels=[0, 0], invert=inv)
            n = int(masks.max())
            prob = 1 / (1 + np.exp(-flows[2]))
            conf = float(np.mean(ndi.mean(prob, labels=masks, index=np.arange(1, n + 1)))) if n else float("nan")
            area = float(np.median(np.bincount(masks.ravel())[1:])) if n else 0.0
            io.imsave(os.path.join(a.output, os.path.splitext(f)[0] + "_labels.png"), masks.astype(np.uint16), check_contrast=False)
            w.writerow([f, n, round(area, 1), round(conf, 4), inv, round(time.perf_counter() - t, 3)])
            print(f"{f}: {n} nuclei, {time.perf_counter() - t:.2f}s", flush=True)
    print(f"task {task}/{n_tasks}: {len(files)} images in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
