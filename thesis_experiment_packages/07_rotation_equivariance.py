#!/usr/bin/env python3
"""E07: evaluate whether predictions follow known image rotations in acute-angle space."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def acute(x: float) -> float:
    x=x%180.0
    return 180.0-x if x>90.0 else x
def main():
    p=argparse.ArgumentParser(description="E07 known-rotation equivariance")
    p.add_argument("--pairs",type=Path); p.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("results")/"E07_rotation")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.pairs:
        with (a.output_dir/"rotation_pairs_template.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=["parent_id","base_pred_deg","rotation_deg","rotated_pred_deg","variant"]).writeheader()
        print(a.output_dir); return
    rows=list(csv.DictReader(a.pairs.open(encoding="utf-8"))); errs=[]
    for r in rows:
        expected=acute(float(r['base_pred_deg'])+float(r['rotation_deg'])); errs.append(abs(float(r['rotated_pred_deg'])-expected))
    result={"n_pairs":len(errs),"mean_equivariance_error_deg":sum(errs)/len(errs),"max_equivariance_error_deg":max(errs)}
    (a.output_dir/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(a.output_dir)
if __name__=="__main__": main()
