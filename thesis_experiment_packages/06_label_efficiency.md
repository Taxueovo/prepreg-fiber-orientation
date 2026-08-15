# E06: Label-Efficiency Learning Curves

Objective: measure performance as the number of independently labeled parent images changes, and test whether DINOv3 is especially useful in low-label regimes. Use 10%, 25%, 50%, and 100% nested subsets with the same five seeds for each initialization.

Run `python thesis_experiment_packages/06_label_efficiency.py --train-parents train_parents.csv` with a CSV containing `parent_id`. Sample parent images rather than patches, and keep validation and test sets fixed.

Plot training-parent fraction against parent-level MAE with a mean and 95% confidence interval at each fraction. Claim reduced label demand only when the low-label difference is stable across seeds.
