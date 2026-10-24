"""Measure each training image's colour, brightness and nucleus size: the inputs to the image groups."""
import os
import numpy as np
import pandas as pd
from skimage import io
import nuclei

rows = []
for i in nuclei.image_ids():
    rgb = nuclei.load_image(i).astype(float)
    colour = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
    masks = sorted(os.listdir(f"{nuclei.ROOT}/{i}/masks"))
    area = float(np.median([(io.imread(f"{nuclei.ROOT}/{i}/masks/{m}") > 0).sum() for m in masks]))
    rows.append({"id": i, "h": rgb.shape[0], "w": rgb.shape[1], "colour": colour,
                 "median_intensity": float(np.median(rgb.mean(-1))), "n_nuclei": len(masks), "median_area": area})
pd.DataFrame(rows).to_csv("data/image_stats.csv", index=False)
print(f"measured {len(rows)} images")
