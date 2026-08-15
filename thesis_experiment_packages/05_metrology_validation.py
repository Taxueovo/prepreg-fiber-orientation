#!/usr/bin/env python3
"""E05: analyze repeated reference-angle measurements for bias and repeatability."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="E05 计量验证")
    p.add_argument("--measurements",type=Path); p.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("results")/"E05_metrology")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.measurements:
        with (a.output_dir/"measurement_template.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=["reference_angle_deg","operator","repeat","measured_angle_deg","image_id","session"]).writeheader()
        print(a.output_dir); return
    rows=list(csv.DictReader(a.measurements.open(encoding="utf-8"))); errors=[float(r['measured_angle_deg'])-float(r['reference_angle_deg']) for r in rows]
    by=defaultdict(list)
    for r,e in zip(rows,errors): by[r['reference_angle_deg']].append(e)
    summary={"n":len(rows),"bias_deg":statistics.mean(errors),"repeatability_sd_deg":statistics.stdev(errors) if len(errors)>1 else 0.0,
      "rmse_deg":math.sqrt(sum(e*e for e in errors)/len(errors)),"bias_by_reference_deg":{k:sum(v)/len(v) for k,v in by.items()}}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(a.output_dir)
if __name__=="__main__": main()
