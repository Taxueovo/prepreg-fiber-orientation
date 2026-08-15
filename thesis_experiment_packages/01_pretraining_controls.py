#!/usr/bin/env python3
"""E01: directly run locked pre-training comparisons with DinoConv.py."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from experiment_runtime import run_training, sha256

SEEDS = [42, 2024, 3407, 5179, 9001]
VARIANTS = ["random_init", "imagenet_supervised", "dinov2", "dinov3"]

def main() -> None:
    p = argparse.ArgumentParser(description="E01 预训练对照（直接训练）")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results") / "E01_pretraining")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--weights", type=Path, help="ImageNet/DINOv2/DINOv3 对照所需的本地权重")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--device", default="cuda:0", help="用 cpu 可显式在CPU运行（很慢）")
    p.add_argument("--seeds", default=",".join(map(str,SEEDS)))
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    require=args.variant != "random_init"
    if require and args.weights is None: p.error("该初始化需要 --weights /path/to/checkpoint.pth")
    manifest = {"experiment": "E01_pretraining", "seeds": seeds, "variant": args.variant,
                "rule": "所有变体使用相同母图划分、训练预算、增强和验证集选参；测试集只在冻结后评估。",
                "weights_file": args.weights.name if args.weights else None,
                "weights_sha256": sha256(args.weights) if args.weights else None,
                "required_artifacts": ["train_params.json", "best_model.pth", "best_test_parent_predictions.csv"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for seed in seeds:
        run=run_training(run_dir=args.output_dir / args.variant, seed=seed, epochs=args.epochs,
            weights=args.weights, require_weights=require, use_attention=True, use_fft=True,
            use_uncertainty=True, device=args.device)
        print(f"完成 {args.variant} seed={seed}: {run}")
if __name__ == "__main__": main()
