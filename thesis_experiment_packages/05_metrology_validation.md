# E05: Reference Metrology and Annotation Repeatability

Objective: quantify reference-value bias and repeatability before interpreting small model differences. When label uncertainty is comparable with model error, it must be treated as part of the measurement system rather than as background context.

Acquire repeated measurements at multiple known angles, preferably across operators and acquisition sessions. Run `python thesis_experiment_packages/05_metrology_validation.py` to create the template, then pass the completed file with `--measurements` to compute bias, repeatability standard deviation, and RMSE.

Report bias versus reference angle, repeatability and reproducibility, a Bland-Altman analysis, and model error relative to reference uncertainty.
