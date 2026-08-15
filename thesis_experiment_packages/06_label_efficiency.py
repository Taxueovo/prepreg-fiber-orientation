#!/usr/bin/env python3
"""E06: create nested parent-image subsets for label-efficiency experiments."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

FRACTIONS=[0.10,0.25,0.50,1.00]; SEEDS=[42,2024,3407,5179,9001]
def main():
    p=argparse.ArgumentParser(description="E06 标注效率模板")
    p.add_argument("--train-parents",type=Path,help="含 parent_id 的冻结训练母图CSV")
    p.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("results")/"E06_label_efficiency")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={"experiment":"E06_label_efficiency","fractions":FRACTIONS,"seeds":SEEDS,"rule":"各子集只从冻结训练母图抽样；验证集、测试集与增强/预算保持不变。"}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if a.train_parents:
        rows=list(csv.DictReader(a.train_parents.open(encoding="utf-8"))); ids=sorted({r['parent_id'] for r in rows})
        for seed in SEEDS:
            rng=random.Random(seed); shuffled=ids[:]; rng.shuffle(shuffled)
            for frac in FRACTIONS:
                chosen=set(shuffled[:max(1,round(len(ids)*frac))]); out=a.output_dir/f"parents_{int(frac*100):03d}_seed{seed}.csv"
                with out.open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=["parent_id"]).writeheader(); [f.write(f"{x}\n") for x in sorted(chosen)]
    with (a.output_dir/"results_template.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=["fraction","seed","variant","n_train_parents","parent_mae_deg","parent_rmse_deg"]).writeheader()
    print(a.output_dir)
if __name__=="__main__": main()
