# E07: Known-Rotation Equivariance

Objective: test whether predictions change consistently with known rotations of the input image. Because the target uses an acute `[0°, 90°]` representation, expected rotated angles must first be folded into acute-angle space.

Run `python thesis_experiment_packages/07_rotation_equivariance.py` to create a template. Record the base prediction, applied rotation, and rotated prediction for multiple copies of the same parent image or patch, then analyze the completed file with `--pairs`. Do not choose rotations or augmentation ranges using test performance.

Report equivariance error by rotation angle. This diagnostic does not replace external-domain testing, but it can expose geometric fragility.
