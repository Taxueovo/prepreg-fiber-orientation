#!/usr/bin/env python3
"""E08: summarize endpoint latency samples without conflating it with acquisition time."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path
def percentile(xs,q):
    xs=sorted(xs); i=(len(xs)-1)*q; lo=int(i); hi=min(lo+1,len(xs)-1); return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
def main():
    p=argparse.ArgumentParser(description="E08 inference efficiency and system latency")
    p.add_argument("--samples",type=Path); p.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("results")/"E08_efficiency")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.samples:
        with (a.output_dir/"latency_template.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=["run_id","stage","latency_ms","peak_vram_mb","hardware","software","batch_size","warmup_done"]).writeheader()
        print(a.output_dir); return
    rows=list(csv.DictReader(a.samples.open(encoding="utf-8"))); vals=[float(r['latency_ms']) for r in rows if r['stage']=='end_to_end']
    if not vals: raise ValueError("At least one row with stage=end_to_end is required")
    out={"n":len(vals),"mean_ms":statistics.mean(vals),"p50_ms":percentile(vals,.5),"p95_ms":percentile(vals,.95),"throughput_img_s":1000/statistics.mean(vals),"max_vram_mb":max(float(r['peak_vram_mb']) for r in rows if r['peak_vram_mb'])}
    (a.output_dir/"summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(a.output_dir)
if __name__=="__main__": main()
