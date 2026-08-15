# E02: Factorial Component Ablation

Objective: evaluate orientation-aware attention, the FFT branch, and heteroscedastic loss as three binary factors in a complete `2 × 2 × 2` design. A small set of single-run snapshots cannot establish component synergy.

Example: `python thesis_experiment_packages/02_factorial_ablation.py --attention 1 --fft 0 --uncertainty 1 --weights /path/to/dinov3.pth`. Run five seeds for each configuration and report parent-level MAE and RMSE. Freeze hyperparameters using the validation set only; do not tune FFT radii, loss weights, or stopping criteria separately on test results.

Report main effects and interactions. If a component improves one configuration but degrades another, describe the interaction directly rather than assuming additive benefits.
