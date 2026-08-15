# Supplementary Experiment Suite

This directory contains eight reproducibility protocols, E01 through E08. Each protocol includes a Python script and a short methods note. Scripts create templates or analyze explicitly supplied CSV files under their own `results/` directory; they do not overwrite the dataset, model weights, or existing experiment records.

Use one frozen parent-image split for every formal comparison. Generate the protocol manifest or CSV template first, preserve the configuration, random seed, checkpoint hash, and per-patch predictions for every run, and then run the analysis step. The test set must never be used for model or hyperparameter selection.

| Protocol | Focus | Primary question |
|---|---|---|
| E01 | Initialization controls | Does DINOv3 initialization reduce label requirements? |
| E02 | Factorial ablation | Do the proposed components make stable independent contributions? |
| E03 | External domains | Does the frozen model generalize across batches and imaging conditions? |
| E04 | Uncertainty | Is uncertainty calibrated and useful for selective prediction? |
| E05 | Metrology | Are the reference values and annotations sufficiently reliable? |
| E06 | Label efficiency | How does performance vary with the number of labeled parent images? |
| E07 | Rotation equivariance | Do predictions follow known image rotations? |
| E08 | System efficiency | What are the measured latency and throughput under a fixed protocol? |
