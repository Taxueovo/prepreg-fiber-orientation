# E01: Pretraining Initialization Controls

Objective: test whether self-supervised DINOv3 initialization outperforms random, supervised ImageNet, and DINOv2 initialization under an otherwise fixed protocol. A DINOv3-only prototype cannot establish a causal pretraining benefit.

Example: `python thesis_experiment_packages/01_pretraining_controls.py --variant dinov3 --weights /path/to/dinov3.pth`. Use `--variant random_init` without `--weights` for the random control. Run five seeds per initialization while keeping inputs, splits, training budget, early stopping, augmentation, FFT settings, and attention settings fixed. Do not use the test set to select learning rates, frequency bands, or epochs.

Aggregate patch predictions by parent image. Report parent-level MAE, RMSE, 95% cluster-bootstrap confidence intervals, and paired error differences relative to random initialization. If an interval crosses zero, report that the difference is not stable.
