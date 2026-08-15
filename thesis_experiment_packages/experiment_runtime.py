"""Reusable, executable bridge from thesis experiment scripts to DinoConv.py.

It intentionally refuses a missing requested checkpoint instead of allowing the
original prototype's silent random-initialisation fallback.
"""
from __future__ import annotations
import csv
import hashlib
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def parent_id(filename: str) -> str:
    match=re.search(r"src([^_./\\]+)", Path(filename).name)
    if not match: raise ValueError(f"无法从图块文件名提取母图ID: {filename}")
    return match.group(1)

def aggregate_parent_predictions(detail_csv: Path, output_csv: Path) -> None:
    with detail_csv.open(encoding="utf-8") as source:
        rows=list(csv.DictReader(source))
    groups={}
    for r in rows:
        key=parent_id(r["filename"]); groups.setdefault(key, []).append(r)
    with output_csv.open("w",newline="",encoding="utf-8") as f:
        fields=["parent_id","y_true_deg","y_pred_deg","pred_std_deg","n_patches"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for key,rs in sorted(groups.items()):
            w.writerow({"parent_id":key,
                "y_true_deg":sum(float(r['true_angle_deg']) for r in rs)/len(rs),
                "y_pred_deg":sum(float(r['pred_angle_deg']) for r in rs)/len(rs),
                "pred_std_deg":sum(float(r['pred_std_deg']) for r in rs)/len(rs),"n_patches":len(rs)})

def run_training(*, run_dir: Path, seed: int, epochs: int, weights: Path|None,
                 require_weights: bool, use_attention: bool, use_fft: bool,
                 use_uncertainty: bool, device: str) -> Path:
    """Run one full DinoConv training job and return its newly-created run directory."""
    import sys
    sys.path.insert(0, str(PROJECT))
    import DinoConv as dc
    if require_weights and (weights is None or not weights.is_file()):
        raise FileNotFoundError("此对照要求 --weights 指向存在的预训练检查点。")
    data_dir=Path(os.environ.get("DINOCONV_DATA_DIR", PROJECT / "database")).expanduser()
    for split in ("train","val","test"):
        csv_path=data_dir / split / f"{split}.csv"
        image_root=data_dir / split / "images"
        if not csv_path.is_file() or not image_root.is_dir():
            raise FileNotFoundError(f"缺少当前项目数据: {csv_path} 或 {image_root}")
        setattr(dc.CFG, f"PATCH_CSV_{split.upper()}", str(csv_path))
        setattr(dc.CFG, f"{split.upper()}_ROOT", str(image_root))
    dc.CFG.SEED=seed; dc.CFG.EPOCHS=epochs; dc.CFG.RUNS_DIR=str(run_dir)
    dc.CFG.USE_ATTENTION=use_attention; dc.CFG.USE_FFT=use_fft; dc.CFG.USE_UNCERTAINTY=use_uncertainty
    dc.CFG.DINOV3_CONVNEXT_PTH=str(weights) if weights else ""; dc.CFG.CONVNEXT_LOCAL_PTH=dc.CFG.DINOV3_CONVNEXT_PTH
    dc.CFG.REQUIRE_PRETRAINED=require_weights; dc.CFG.DEVICE=device; dc.CFG.FORCE_CUDA=(device!="cpu"); dc.CFG.CUDA_DEVICE=device
    dc.train_transform=dc.FiberTrainTransform(blur_p=dc.CFG.TRAIN_BLUR_P)
    before=set(run_dir.glob("FiberAngleNet_DINOv3_*")) if run_dir.exists() else set()
    run_dir.mkdir(parents=True,exist_ok=True); dc.main()
    created=sorted(set(run_dir.glob("FiberAngleNet_DINOv3_*"))-before, key=lambda x:x.stat().st_mtime)
    if not created: raise RuntimeError("训练结束但未找到运行目录。")
    detail=created[-1]/"tables"/"best_test_details.csv"
    if detail.is_file(): aggregate_parent_predictions(detail, created[-1]/"tables"/"best_test_parent_predictions.csv")
    return created[-1]

def run_inference(*, checkpoint: Path, patch_csv: Path, image_root: Path, output_dir: Path,
                  use_attention: bool, use_fft: bool, use_uncertainty: bool, device: str) -> Path:
    """Evaluate a frozen checkpoint on a CSV/image-root pair and emit patch/parent predictions."""
    if not checkpoint.is_file() or not patch_csv.is_file() or not image_root.is_dir():
        raise FileNotFoundError("checkpoint、patch_csv 和 image_root 必须均存在。")
    import sys; sys.path.insert(0, str(PROJECT))
    import torch
    from torch.utils.data import DataLoader
    import DinoConv as dc
    dev=torch.device(device)
    model=dc.FiberAngleNet(local_pth="", use_attention=use_attention, use_fft=use_fft,
                           use_uncertainty=use_uncertainty).to(dev)
    state=torch.load(checkpoint,map_location=dev)
    model.load_state_dict(state,strict=True)
    ds=dc.PatchDataset(str(patch_csv),str(image_root),transform=dc.val_transform)
    loader=DataLoader(ds,batch_size=dc.CFG.BATCH_SIZE,shuffle=False,num_workers=dc.CFG.NUM_WORKERS,
                      pin_memory=(dev.type=="cuda"))
    criterion=dc.AcuteAngleLoss(alpha=dc.CFG.LOSS_ALPHA,use_uncertainty=use_uncertainty,
        min_std_deg=dc.CFG.MIN_STD_DEG,max_std_deg=dc.CFG.MAX_STD_DEG,std_reg_threshold=dc.CFG.STD_REG_THRESHOLD)
    metrics=dc.evaluate(model,loader,dev,criterion,dev.type=="cuda",save_details=True,desc="External")
    output_dir.mkdir(parents=True,exist_ok=True)
    patch_out=output_dir/"patch_predictions.csv"; dc.write_csv(str(patch_out),metrics["details"],list(metrics["details"][0]))
    aggregate_parent_predictions(patch_out,output_dir/"parent_predictions.csv")
    (output_dir/"metrics.json").write_text(__import__('json').dumps({k:v for k,v in metrics.items() if k!='details'},ensure_ascii=False,indent=2),encoding="utf-8")
    return output_dir
