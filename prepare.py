"""Assign every training image to an appearance group and save ground-truth label images."""
import os
import numpy as np
import pandas as pd
import nuclei

from skimage import color
stats = nuclei.assign_groups(pd.read_csv("data/image_stats.csv"))
# same rule segment.py uses to decide when to invert: grey median above 0.4
stats["bright"] = [bool(np.median(color.rgb2gray(nuclei.load_image(i))) > 0.4) for i in stats.id]
stats.to_csv("data/groups.csv", index=False)
os.makedirs("data/truth", exist_ok=True)
for i in stats.id:
    np.savez_compressed(f"data/truth/{i}.npz", lab=nuclei.load_truth(i))
print(stats.group.value_counts().to_string())
print(pd.crosstab(stats.group, stats.bright).to_string())
