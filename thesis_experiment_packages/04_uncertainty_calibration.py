#!/usr/bin/env python3
"""E04: compute basic regression uncertainty metrics from parent-level predictions."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path

Z = {0.50:0.67449, 0.80:1.28155, 0.90:1.64485, 0.95:1.95996}
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def rank(xs):
    order=sorted(range(len(xs)), key=lambda i: xs[i]); out=[0.0]*len(xs); i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[order[j+1]]==xs[order[i]]: j+=1
        r=(i+j+2)/2
        for k in range(i,j+1): out[order[k]]=r
        i=j+1
    return out
def corr(x,y):
    mx,my=mean(x),mean(y); d=math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
    return float("nan") if not d else sum((a-mx)*(b-my) for a,b in zip(x,y))/d
def analyze(path: Path):
    rows=list(csv.DictReader(path.open(encoding="utf-8"))); required={"y_true_deg","y_pred_deg","pred_std_deg"}
    if not rows or not required.issubset(rows[0]): raise ValueError(f"CSV必须包含 {sorted(required)}")
    err=[abs(float(r['y_pred_deg'])-float(r['y_true_deg'])) for r in rows]; std=[max(float(r['pred_std_deg']),1e-9) for r in rows]
    out={"n_parents":len(rows),"mae_deg":mean(err),"rmse_deg":math.sqrt(mean([e*e for e in err])),"spearman_std_abs_error":corr(rank(std),rank(err)),"mpiW_95_deg":mean([2*Z[.95]*s for s in std])}
    for level,z in Z.items(): out[f"picp_{int(level*100)}"] = mean([float(e)<=z*s for e,s in zip(err,std)])
    return out
def main():
    p=argparse.ArgumentParser(description="E04 回归不确定性校准")
    p.add_argument("--predictions",type=Path); p.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("results")/"E04_uncertainty")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.predictions: (a.output_dir/"summary.json").write_text(json.dumps(analyze(a.predictions),ensure_ascii=False,indent=2),encoding="utf-8")
    else:
        with (a.output_dir/"parent_predictions_template.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=["parent_id","y_true_deg","y_pred_deg","pred_std_deg","split" ]).writeheader()
    print(a.output_dir)
if __name__=="__main__": main()
