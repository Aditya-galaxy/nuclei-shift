| Method | fluor_small (n=413) | fluor_large (n=133) | tissue_colour (n=108) | brightfield (n=16) |
|---|---|---|---|---|
| classical | 0.493 [0.469, 0.514] | 0.386 [0.360, 0.408] | 0.183 [0.164, 0.200] | 0.006 [0.004, 0.008] |
| cellpose nuclei, off the shelf | 0.601 [0.520, 0.698] | 0.765 [0.737, 0.796] | 0.000 [0.000, 0.000] | not run |
| cellpose nuclei, inverted when bright | 0.601 [0.520, 0.698] | 0.765 [0.737, 0.796] | 0.000 [0.000, 0.000] | not run |
| cellpose cyto3, off the shelf | not run | not run | not run | not run |
| cellpose cyto3, inverted when bright | not run | not run | not run | not run |

Data Science Bowl score: per image, the mean over IoU thresholds 0.5 to 0.95 of TP / (TP + FP + FN); averaged over images with a 95% bootstrap interval.
