#!/usr/bin/env python3
"""E03: run a frozen checkpoint on a separately registered external domain."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from experiment_runtime import run_inference, sha256

def main() -> None:
    p = argparse.ArgumentParser(description="E03 zero-shot external-domain inference")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results") / "E03_external_domain")
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--patch-csv",type=Path,required=True); p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--attention",type=int,choices=[0,1],required=True); p.add_argument("--fft",type=int,choices=[0,1],required=True); p.add_argument("--uncertainty",type=int,choices=[0,1],required=True)
    p.add_argument("--device",default="cuda:0")
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["domain_id","parent_id","material_batch","capture_date","illumination","camera","distance_mm","used_for_selection","y_true_deg","notes"]
    registry = a.output_dir / "external_parent_registry.csv"
    if not registry.exists():
        with registry.open("w", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=fields).writeheader()
    protocol = {"experiment":"E03_external_domain", "zero_shot_first":True,
      "rule":"Register all external parent images after model selection; used_for_selection must remain false.",
      "minimum_domains":["At least two unseen material batches","Changed illumination or reflectance","Changed device or working distance"],
      "outputs":["Zero-shot parent predictions", "Separately reported optional few-shot adaptation", "Domain-stratified MAE, RMSE, bias, and confidence intervals"]}
    (a.output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output_dir/"checkpoint.sha256").write_text(sha256(a.checkpoint),encoding="utf-8")
    out=run_inference(checkpoint=a.checkpoint,patch_csv=a.patch_csv,image_root=a.image_root,output_dir=a.output_dir/"inference",
        use_attention=bool(a.attention),use_fft=bool(a.fft),use_uncertainty=bool(a.uncertainty),device=a.device)
    print(f"External-domain zero-shot inference completed: {out}")
if __name__ == "__main__": main()
