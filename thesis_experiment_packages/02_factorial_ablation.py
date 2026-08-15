#!/usr/bin/env python3
"""E02: directly run one configuration of the 2×2×2 factorial ablation."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
from experiment_runtime import run_training, sha256

SEEDS = [42, 2024, 3407, 5179, 9001]
FACTORS = ["orientation_attention", "fft_branch", "uncertainty_loss"]

def main() -> None:
    p = argparse.ArgumentParser(description="E02 析因消融（直接训练一个配置）")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results") / "E02_factorial")
    p.add_argument("--attention", type=int, choices=[0,1], required=True)
    p.add_argument("--fft", type=int, choices=[0,1], required=True)
    p.add_argument("--uncertainty", type=int, choices=[0,1], required=True)
    p.add_argument("--weights", type=Path, required=True, help="冻结的 DINOv3 权重")
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default=",".join(map(str,SEEDS)))
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    configs = [{"name": "_".join(f"{k[:3]}{int(v)}" for k,v in zip(FACTORS, values)), **dict(zip(FACTORS, values))}
               for values in itertools.product([False, True], repeat=3)]
    seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    (a.output_dir / "manifest.json").write_text(json.dumps({"experiment":"E02_factorial", "seeds":seeds, "configs":configs,
        "freeze_rule":"FFT r1/r2 和损失权重仅由验证集确定；八个配置共享其余训练条件。"}, ensure_ascii=False, indent=2), encoding="utf-8")
    name=f"att{a.attention}_fft{a.fft}_unc{a.uncertainty}"
    for seed in seeds:
        run=run_training(run_dir=a.output_dir/name,seed=seed,epochs=a.epochs,weights=a.weights,
            require_weights=True,use_attention=bool(a.attention),use_fft=bool(a.fft),
            use_uncertainty=bool(a.uncertainty),device=a.device)
        print(f"完成 {name} seed={seed}: {run}")
if __name__ == "__main__": main()
