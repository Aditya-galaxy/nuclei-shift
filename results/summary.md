| Method | fluor_small (n=415) | fluor_large (n=131) | tissue_colour (n=108) | brightfield (n=16) |
|---|---|---|---|---|
| classical | 0.492 [0.470, 0.515] | 0.385 [0.360, 0.408] | 0.183 [0.164, 0.200] | 0.006 [0.004, 0.008] |
| cellpose nuclei, off the shelf | 0.565 [0.543, 0.587] | 0.736 [0.712, 0.758] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] |
| cellpose nuclei, inverted when bright | 0.565 [0.543, 0.587] | 0.736 [0.712, 0.758] | 0.509 [0.482, 0.536] | 0.454 [0.406, 0.501] |
| cellpose cyto3, off the shelf | 0.636 [0.615, 0.656] | 0.753 [0.729, 0.775] | 0.321 [0.299, 0.344] | 0.004 [0.001, 0.006] |
| cellpose cyto3, inverted when bright | 0.636 [0.615, 0.656] | 0.753 [0.729, 0.775] | 0.504 [0.479, 0.527] | 0.461 [0.409, 0.507] |

Data Science Bowl score: per image, the mean over IoU thresholds 0.5 to 0.95 of TP / (TP + FP + FN); averaged over images with a 95% bootstrap interval.
