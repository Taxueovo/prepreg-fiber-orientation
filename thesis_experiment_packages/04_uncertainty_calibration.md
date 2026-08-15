# E04: Regression Uncertainty and Selective Prediction

Objective: evaluate whether predicted uncertainty is calibrated and useful for ranking risk. A weak uncertainty-error correlation alone does not demonstrate calibration. Use parent images excluded from training and model selection, with fields `parent_id`, `y_true_deg`, `y_pred_deg`, and `pred_std_deg`.

Run `python thesis_experiment_packages/04_uncertainty_calibration.py` to create a template, then analyze predictions with `--predictions parent_predictions.csv`. The script reports MAE, RMSE, Spearman correlation, 50/80/90/95% prediction-interval coverage probability (PICP), and 95% mean prediction-interval width (MPIW). Interpret coverage and interval width together.

Also plot a risk-coverage curve by retaining increasing fractions of samples ranked by predicted standard deviation. Do not claim an effective review policy if rejecting high-risk samples does not materially reduce error.
