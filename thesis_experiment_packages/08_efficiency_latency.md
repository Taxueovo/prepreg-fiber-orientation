# E08: Inference Efficiency and System Latency

Objective: report measured end-to-end performance rather than model-forward time alone. Run `python thesis_experiment_packages/08_efficiency_latency.py` to generate a template. Under fixed hardware and software, batch size 1, and a documented warm-up, record loading, preprocessing, inference, output writing, and end-to-end latency.

Analyze the completed CSV with `--samples latency.csv` to obtain mean, P50, P95, throughput, and peak memory. Acquisition, transfer, and human review must be reported separately if they are not included in the end-to-end samples.

Document hardware, software, image size, sample count, latency distribution, throughput, memory, parameter count, and FLOPs. Do not extrapolate to production-line readiness without a real integrated-system evaluation.
