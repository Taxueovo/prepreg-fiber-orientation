# Dataset — Carbon-Fiber Prepreg Surface Patches

De-identified sample patches from the dataset behind this project. The **full dataset is not committed** to this repository; a few representative patches are included for illustration so the data format and appearance are visible without downloading ~7.6 GB.

## `samples/`

Six 512×512 RGB microscopy patches sampled at quantiles of the training orientation-angle distribution:

| File | Orientation angle |
|---|---|
| `sample_1_angle41.0.jpg` | ≈ 41.0° |
| `sample_2_angle57.9.jpg` | ≈ 57.9° |
| `sample_3_angle68.2.jpg` | ≈ 68.2° |
| `sample_4_angle79.0.jpg` | ≈ 79.0° |
| `sample_5_angle86.2.jpg` | ≈ 86.2° |
| `sample_6_angle88.2.jpg` | ≈ 88.2° |

## Full dataset (not included)

| Split | Patches | Manifest |
|---|---|---|
| train | 83,034 | `train.csv` |
| val | 10,332 | `val.csv` |
| test | 10,458 | `test.csv` |

≈ 104,000 patches in total (~7.6 GB). Each patch is a 512×512 RGB JPEG cropped from a parent microscopy image; labels are acute surface-fiber orientation angles in `[0°, 90°]`.

Manifest format:

```csv
patch_filename,source_image,angle,x,y
train_000000_srcPTAX0535_x0_y0.jpg,PTAX0535.JPG,78.1116,0,0
```

- `patch_filename` — file name relative to the split's `images/` directory
- `source_image` — parent image the patch was cropped from
- `angle` — ground-truth acute orientation angle
- `x`, `y` — crop offset within the source image

## Access

The full dataset is available on request — open an issue or reach out via GitHub.

## License

Data and label manifests are **not** covered by the repository's MIT license.
