# E03: External-Domain and Imaging Robustness

Objective: distinguish internal testing from genuine domain transfer. Run `python thesis_experiment_packages/03_external_domain.py --checkpoint best_model.pth --patch-csv external.csv --image-root /path/to/external-images --attention 1 --fft 1 --uncertainty 1`. The script exports patch- and parent-level predictions and creates an external-domain registry for batch, date, illumination, device, and working-distance metadata.

The first external evaluation must be zero-shot: no fine-tuning, threshold adjustment, or model selection on external data. Report any later few-shot adaptation separately. For each domain, report the number of parent images, MAE, RMSE, bias, a 95% confidence interval, and representative failure cases.

Present internal, zero-shot external, and adapted results as separate result groups.
