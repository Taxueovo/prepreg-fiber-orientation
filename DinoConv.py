import argparse
import os
import csv
import math
import json
import time
import random
import datetime
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# ===================== Config =====================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    os.environ.get("DINOCONV_DATA_DIR", PROJECT_ROOT / "database")
).expanduser()
DEFAULT_WEIGHTS = os.environ.get("DINOCONV_WEIGHTS", "")


@dataclass
class Config:
    PATCH_CSV_TRAIN: str = str(DEFAULT_DATA_DIR / "train" / "train.csv")
    PATCH_CSV_VAL: str = str(DEFAULT_DATA_DIR / "val" / "val.csv")
    PATCH_CSV_TEST: str = str(DEFAULT_DATA_DIR / "test" / "test.csv")

    TRAIN_ROOT: str = str(DEFAULT_DATA_DIR / "train" / "images")
    VAL_ROOT: str = str(DEFAULT_DATA_DIR / "val" / "images")
    TEST_ROOT: str = str(DEFAULT_DATA_DIR / "test" / "images")

    DINOV3_CONVNEXT_PTH: str = DEFAULT_WEIGHTS
    CONVNEXT_LOCAL_PTH: str = DEFAULT_WEIGHTS

    FORCE_CUDA: bool = os.environ.get("DINOCONV_FORCE_CUDA", "0") == "1"
    CUDA_DEVICE: str = "cuda:0"
    DEVICE: str = os.environ.get("DINOCONV_DEVICE", "")

    EPOCHS: int = 150
    WARMUP_EPOCHS: int = 5
    LR: float = 5e-4
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    MIXED_PRECISION: bool = True
    EARLY_STOP_PATIENCE: int = 15

    LOSS_ALPHA: float = 0.6
    USE_ATTENTION: bool = True
    USE_FFT: bool = True
    USE_UNCERTAINTY: bool = True
    REQUIRE_PRETRAINED: bool = False
    MIN_STD_DEG: float = 0.25
    MAX_STD_DEG: float = 5.0
    STD_REG_THRESHOLD: float = 3.0

    EPS_NORM: float = 1e-8
    CLIP_GRAD_NORM: float = 1.0
    SEED: int = 42
    RUNS_DIR: str = os.environ.get("DINOCONV_RUNS_DIR", "runs")

    FFT_R1_RATIO: float = 0.05
    FFT_R2_RATIO: float = 0.40
    FFT_POOL_SIZE: int = 8

    GABOR_NUM_ORI: int = 8
    TRAIN_BLUR_P: float = 0.3


CFG = Config()


# ===================== Utils =====================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_join(root: str, rel_path: str) -> str:
    """Resolve a dataset member while preventing absolute/path-traversal input."""
    root_path = Path(root).expanduser().resolve()
    normalized = rel_path.replace("\\", "/")
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise ValueError(f"Dataset path must be relative: {rel_path!r}")
    candidate = (root_path / normalized).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes image root: {rel_path!r}") from exc
    return str(candidate)


def manifest_path(path_value: str) -> str:
    """Return a reproducibility-friendly path without exposing a home directory."""
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (OSError, ValueError):
        return path.name


def public_config_snapshot(config: Config) -> Dict:
    snapshot = asdict(config)
    for key in (
        "PATCH_CSV_TRAIN", "PATCH_CSV_VAL", "PATCH_CSV_TEST",
        "TRAIN_ROOT", "VAL_ROOT", "TEST_ROOT",
        "DINOV3_CONVNEXT_PTH", "CONVNEXT_LOCAL_PTH", "RUNS_DIR",
    ):
        snapshot[key] = manifest_path(str(snapshot[key]))
    return snapshot


def fold_to_acute_deg(angle_deg: float) -> float:
    a = angle_deg % 180.0
    if a > 90.0:
        a = 180.0 - a
    return a


def angle_mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def angle_rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ===================== Dataset =====================
class PatchDataset(Dataset):
    def __init__(self, csv_file: str, data_root: str, transform=None):
        self.samples: List[Tuple[str, float, str]] = []
        self.transform = transform

        missing = 0
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sample_id = row["patch_filename"].replace("\\", "/")
                full_path = safe_join(data_root, sample_id)
                if not os.path.exists(full_path):
                    missing += 1
                    continue

                raw_angle = float(row["angle"])
                acute_angle = fold_to_acute_deg(raw_angle)
                self.samples.append((full_path, acute_angle, sample_id))

        if not self.samples:
            raise ValueError(f"No valid samples: root={data_root}, csv={csv_file}")
        if missing:
            print(f"[Warning] {missing} files in {csv_file} not found (skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, acute_angle, sample_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        label = torch.tensor(acute_angle, dtype=torch.float32)
        if self.transform:
            out = self.transform(img, label)
            img, label = out if isinstance(out, tuple) else (out, label)
        # Never write a workstation's absolute dataset path into prediction CSVs.
        return img, label, sample_id


# ===================== Transforms =====================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FiberTrainTransform:
    def __init__(self, out_size: int = 224, blur_p: float = 0.3):
        self.blur_p = blur_p
        self.color = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.15, hue=0.05
        )
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.crop = transforms.RandomResizedCrop(out_size, scale=(0.75, 1.0))

    def __call__(self, img: Image.Image, label: torch.Tensor):
        img = self.color(img)
        if random.random() < self.blur_p:
            img = self.blur(img)
        img = self.crop(img)
        img = TF.to_tensor(img)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        return img, label


class ImgOnlyTransform:
    def __init__(self, t):
        self.t = t

    def __call__(self, img, label):
        return self.t(img), label


_val_test_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_transform = FiberTrainTransform(blur_p=CFG.TRAIN_BLUR_P)
val_transform = ImgOnlyTransform(_val_test_t)
test_transform = ImgOnlyTransform(_val_test_t)


# ===================== Model Modules =====================
class FFTOrientationModule(nn.Module):
    def __init__(self, r1_ratio: float = 0.05, r2_ratio: float = 0.40,
                 pool_size: int = 8):
        super().__init__()
        self.r1_ratio = r1_ratio
        self.r2_ratio = r2_ratio
        self.pool_size = pool_size
        self.proj = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.out_dim = 8 * pool_size * pool_size
        self._cached_mask = None
        self._cached_hw = None

    def _make_ring_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        cy, cx = H // 2, W // 2
        r1 = int(min(H, W) * self.r1_ratio)
        r2 = int(min(H, W) * self.r2_ratio)
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W, device=device).float() - cx
        dist = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        mask = ((dist >= r1) & (dist <= r2)).float()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        mag = torch.abs(torch.fft.fftshift(fft))
        H, W = mag.shape[-2:]
        if self._cached_hw != (H, W):
            self._cached_mask = self._make_ring_mask(H, W, x.device)
            self._cached_hw = (H, W)
        filtered = mag * self._cached_mask
        feat = self.proj(filtered)
        feat = F.adaptive_avg_pool2d(feat, self.pool_size)
        return feat.flatten(1)


class GaborAttention(nn.Module):
    """
    可学习方向注意力模块。
    [FIXED-B] 插入 ConvNeXt-Tiny stage2 (timm 索引 [2]) 之后，该处通道数为 384。
    """

    def __init__(self, channels: int = 384, num_orientations: int = 8):  # [FIXED-B] 384
        super().__init__()
        self.ori_weights = nn.Parameter(
            torch.ones(num_orientations) / num_orientations
        )
        self.dir_conv = nn.Conv2d(
            channels, num_orientations,
            kernel_size=7, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(num_orientations)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_orientations, num_orientations * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_orientations * 2, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ori_feat = F.relu(self.bn(self.dir_conv(x)))
        w = F.softmax(self.ori_weights, dim=0).view(1, -1, 1, 1)
        attn_map = (ori_feat * w).sum(dim=1, keepdim=True)
        pooled = self.gap(ori_feat).flatten(1)
        ch_attn = self.fc(pooled).view(B, C, 1, 1)
        return x * torch.sigmoid(attn_map) * ch_attn


# ===================== Main Model =====================
class FiberAngleNet(nn.Module):
    """
    [FIXED-B] 手动走 backbone 各 stage，将 Gabor 注意力真正插入 stage2 之后，
              使其对 stage3/stage4 的输出产生实际影响。
    """

    def __init__(self, local_pth: str = "",
                 num_orientations: int = 8,
                 fft_r1: float = 0.05, fft_r2: float = 0.40,
                 fft_pool: int = 8,
                 use_attention: bool = True,
                 use_fft: bool = True,
                 use_uncertainty: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.use_fft = use_fft
        self.use_uncertainty = use_uncertainty

        try:
            import timm
        except ImportError:
            raise ImportError("请安装 timm 库: pip install timm")

        # [FIXED-B] 不再使用 features_only，而是保留完整模型，以便手动调用各 stage
        backbone = timm.create_model(
            'convnext_tiny',
            pretrained=False,
            num_classes=0,       # 去掉分类头
            global_pool='',      # 关闭全局池化,后面我们自己做
        )

        # 加载 DINOv3 预训练权重
        if local_pth and os.path.exists(local_pth):
            print(f"[Backbone] Loading DINOv3-ConvNeXt from {local_pth}")
            state_dict = torch.load(local_pth, map_location="cpu")

            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith("backbone."):
                    name = name[9:]
                elif name.startswith("module."):
                    name = name[7:]
                elif name.startswith("model."):
                    name = name[6:]
                new_state_dict[name] = v

            msg = backbone.load_state_dict(new_state_dict, strict=False)
            print(
                f"[Backbone] Loaded weights. Missing: {len(msg.missing_keys)}, "
                f"Unexpected: {len(msg.unexpected_keys)}"
            )
        elif CFG.REQUIRE_PRETRAINED:
            raise FileNotFoundError(
                "预训练对照要求指定有效权重；当前未找到 DINOV3_CONVNEXT_PTH。"
            )
        else:
            print("[Backbone] WARNING: DINOv3 weights not found, using random initialization")

        self.backbone = backbone

        # [FIXED-B] 动态确认 stage2 的输出通道,避免硬编码错误
        # timm 的 ConvNeXt-Tiny: [96, 192, 384, 768]
        stage2_channels = self._infer_stage_channels(backbone, stage_idx=2, default=384)
        last_channels = getattr(backbone, "num_features", 768)
        print(f"[Backbone] stage2 channels = {stage2_channels}, "
              f"last channels = {last_channels}")

        # [FIXED-B] Gabor 用实际的 stage2 通道数实例化
        self.gabor = (GaborAttention(channels=stage2_channels, num_orientations=num_orientations)
                      if use_attention else nn.Identity())

        self.fft_module = (FFTOrientationModule(r1_ratio=fft_r1, r2_ratio=fft_r2, pool_size=fft_pool)
                           if use_fft else None)

        # [FIXED-B] 方案 B 下 Gabor 是串行注意力,不并入 fusion,
        #           因此 fused_dim 仍然是 last_channels + fft_dim
        fused_dim = last_channels + (self.fft_module.out_dim if self.fft_module else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.head_angle = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _infer_stage_channels(backbone: nn.Module, stage_idx: int, default: int) -> int:
        """[FIXED-B] 优先通过 feature_info 读取,失败则返回默认值."""
        try:
            return int(backbone.feature_info.channels()[stage_idx])
        except Exception:
            return default

    def _forward_backbone_with_gabor(self, x: torch.Tensor) -> torch.Tensor:
        """
        [FIXED-B] 手动遍历 backbone,将 Gabor 插入 stage2 之后。
        timm ConvNeXt forward_features 等价路径: stem -> stages -> norm_pre
        """
        x = self.backbone.stem(x)
        x = self.backbone.stages[0](x)   # 96
        x = self.backbone.stages[1](x)   # 192
        x = self.backbone.stages[2](x)   # 384
        x = self.gabor(x)                # 可由 USE_ATTENTION 关闭以形成匹配消融
        x = self.backbone.stages[3](x)   # 768
        # norm_pre 在 timm 里要么是 LayerNorm,要么是 Identity,都安全可调
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_raw = x

        cnn_feat_map = self._forward_backbone_with_gabor(x)            # [B, 768, H, W]
        cnn_feat = F.adaptive_avg_pool2d(cnn_feat_map, 1).flatten(1)   # [B, 768]

        if self.fft_module is not None:
            fft_feat = self.fft_module(x_raw)                          # [B, fft_dim]
            fused_input = torch.cat([cnn_feat, fft_feat], dim=1)
        else:
            fused_input = cnn_feat
        fused = self.fusion(fused_input)

        pred_angle_deg = 90.0 * torch.sigmoid(self.head_angle(fused).squeeze(1))

        if self.use_uncertainty:
            log_var = self.head_logvar(fused).squeeze(1)
        else:
            log_var = None
        return pred_angle_deg, log_var


# ===================== Loss =====================
class AcuteAngleLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.6,
                 use_uncertainty: bool = True,
                 min_std_deg: float = 0.25,
                 max_std_deg: float = 5.0,
                 std_reg_threshold: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.use_uncertainty = use_uncertainty
        self.min_log_var = math.log(min_std_deg ** 2)
        self.max_log_var = math.log(max_std_deg ** 2)
        self.reg_log_var = math.log(std_reg_threshold ** 2)

    def forward(self,
                pred_angle_deg: torch.Tensor,
                log_var: Optional[torch.Tensor],
                target_angle_deg: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        err = pred_angle_deg - target_angle_deg
        abs_err = err.abs()
        sq_err = err.pow(2)
        mae_loss = abs_err.mean()
        rmse_deg = torch.sqrt(sq_err.mean() + 1e-12)

        if self.use_uncertainty and log_var is not None:
            log_var_c = log_var.clamp(self.min_log_var, self.max_log_var)
            nll_loss = 0.5 * (sq_err * torch.exp(-log_var_c) + log_var_c).mean()
            var_reg = F.relu(log_var_c - self.reg_log_var).mean()
            total = self.alpha * nll_loss + (1.0 - self.alpha) * mae_loss + 0.1 * var_reg
            mean_logvar = log_var_c.mean().item()
            mean_std_deg = torch.exp(0.5 * log_var_c).mean().item()
        else:
            log_var_c = None
            nll_loss = torch.zeros((), device=pred_angle_deg.device)
            var_reg = torch.zeros((), device=pred_angle_deg.device)
            total = mae_loss
            mean_logvar = 0.0
            mean_std_deg = 0.0

        return total, {
            "nll_loss": nll_loss.item(),
            "mae_loss": mae_loss.item(),
            "var_reg": var_reg.item(),
            "angle_mae": mae_loss.item(),
            "angle_rmse": rmse_deg.item(),
            "mean_logvar": mean_logvar,
            "mean_std_deg": mean_std_deg,
        }


# ===================== Evaluate =====================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: AcuteAngleLoss, use_amp: bool,
             save_details: bool = False, desc: str = "Eval") -> dict:
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    all_logvars_c = []
    details = []

    for images, labels, paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_angle_deg, log_var = model(images)
            loss, _ = criterion(pred_angle_deg, log_var, labels)

        total_loss += loss.item() * images.size(0)

        pred_np = pred_angle_deg.detach().cpu().numpy()
        true_np = labels.detach().cpu().numpy()
        all_pred.append(pred_np)
        all_true.append(true_np)

        if criterion.use_uncertainty and log_var is not None:
            lv_c = log_var.clamp(criterion.min_log_var, criterion.max_log_var).detach().cpu().numpy()
            all_logvars_c.append(lv_c)
            std_deg = np.exp(0.5 * lv_c)
        else:
            std_deg = np.zeros_like(pred_np)

        if save_details:
            err_np = np.abs(pred_np - true_np)
            for i in range(len(paths)):
                details.append({
                    "filename": paths[i],
                    "true_angle_deg": float(true_np[i]),
                    "pred_angle_deg": float(pred_np[i]),
                    "angle_error_deg": float(err_np[i]),
                    "pred_std_deg": float(std_deg[i]),
                })

    n = len(loader.dataset)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    abs_err = np.abs(y_pred - y_true)

    if len(all_logvars_c) > 0:
        lv_c = np.concatenate(all_logvars_c)
        mean_logvar = float(np.mean(lv_c))
        mean_std_deg = float(np.mean(np.exp(0.5 * lv_c)))
    else:
        mean_logvar = 0.0
        mean_std_deg = 0.0

    return {
        "loss": total_loss / n,
        "angle_mae_deg": float(np.mean(abs_err)),
        "angle_rmse_deg": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "angle_r2": r2_score_np(y_true, y_pred),
        "mean_logvar": mean_logvar,
        "mean_std_deg": mean_std_deg,
        "details": details,
    }


# ===================== Train One Epoch =====================
def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    criterion: AcuteAngleLoss, optimizer: optim.Optimizer,
                    scaler, use_amp: bool, clip_grad_norm: float) -> dict:
    model.train()
    total_loss = total_mae = total_rmse2 = 0.0
    n = 0
    grad_norm_sum = 0.0
    aux_sums = {
        "nll_loss": 0.0,
        "mae_loss": 0.0,
        "var_reg": 0.0,
        "mean_logvar": 0.0,
        "mean_std_deg": 0.0,
    }

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_angle_deg, log_var = model(images)
            loss, aux = criterion(pred_angle_deg, log_var, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if clip_grad_norm > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            gn_value = float(gn)
        else:
            gn_value = 0.0

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        bs = images.size(0)
        with torch.no_grad():
            err = (pred_angle_deg - labels).abs().detach().cpu().numpy()

        total_loss += loss.item() * bs
        total_mae += err.sum()
        total_rmse2 += (err ** 2).sum()
        grad_norm_sum += gn_value * bs
        n += bs

        for k in aux_sums:
            aux_sums[k] += aux.get(k, 0.0) * bs

    return {
        "loss": total_loss / n,
        "angle_mae_deg": float(total_mae / n),
        "angle_rmse_deg": math.sqrt(total_rmse2 / n),
        "grad_norm": grad_norm_sum / n,
        **{k: v / n for k, v in aux_sums.items()},
    }


# ===================== Main =====================
def main():
    set_seed(CFG.SEED)

    device = (torch.device(CFG.DEVICE) if CFG.DEVICE else
              (torch.device(CFG.CUDA_DEVICE) if CFG.FORCE_CUDA else
               torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    if CFG.FORCE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    use_amp = CFG.MIXED_PRECISION and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = datetime.datetime.now().strftime("FiberAngleNet_DINOv3_%Y%m%d_%H%M%S")
    save_dir = os.path.join(CFG.RUNS_DIR, run_name)
    ensure_dir(save_dir)
    ensure_dir(os.path.join(save_dir, "tables"))

    with open(os.path.join(save_dir, "train_params.json"), "w", encoding="utf-8") as f:
        json.dump(public_config_snapshot(CFG), f, indent=2, ensure_ascii=False)

    print("Loading datasets...")
    train_ds = PatchDataset(CFG.PATCH_CSV_TRAIN, CFG.TRAIN_ROOT, transform=train_transform)
    val_ds = PatchDataset(CFG.PATCH_CSV_VAL, CFG.VAL_ROOT, transform=val_transform)
    test_ds = PatchDataset(CFG.PATCH_CSV_TEST, CFG.TEST_ROOT, transform=test_transform)

    persistent = CFG.NUM_WORKERS > 0
    loader_kw = dict(
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        persistent_workers=persistent,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    print(f"Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    model = FiberAngleNet(
        local_pth=CFG.DINOV3_CONVNEXT_PTH,
        num_orientations=CFG.GABOR_NUM_ORI,
        fft_r1=CFG.FFT_R1_RATIO,
        fft_r2=CFG.FFT_R2_RATIO,
        fft_pool=CFG.FFT_POOL_SIZE,
        use_attention=CFG.USE_ATTENTION,
        use_fft=CFG.USE_FFT,
        use_uncertainty=CFG.USE_UNCERTAINTY,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.2f} M")

    criterion = AcuteAngleLoss(
        alpha=CFG.LOSS_ALPHA,
        use_uncertainty=CFG.USE_UNCERTAINTY,
        min_std_deg=CFG.MIN_STD_DEG,
        max_std_deg=CFG.MAX_STD_DEG,
        std_reg_threshold=CFG.STD_REG_THRESHOLD,
    )

    # [FIXED-B] 精确按 "backbone.*" 前缀分组,避免原来模糊匹配误伤 head_*/fusion.norm
    backbone_params, new_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            new_params.append(p)
    print(f"[Optim] backbone params: {sum(p.numel() for p in backbone_params)/1e6:.2f} M | "
          f"new params: {sum(p.numel() for p in new_params)/1e6:.2f} M")

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": CFG.LR * 0.1},
        {"params": new_params, "lr": CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    if CFG.WARMUP_EPOCHS > 0 and CFG.WARMUP_EPOCHS < CFG.EPOCHS:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.2, end_factor=1.0, total_iters=CFG.WARMUP_EPOCHS
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.EPOCHS - CFG.WARMUP_EPOCHS, eta_min=CFG.MIN_LR
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[CFG.WARMUP_EPOCHS]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    best_val_mae = float("inf")
    best_epoch = -1
    patience = 0
    history: List[Dict] = []
    t0 = time.time()

    print(f"Device={device} | AMP={use_amp} | Epochs={CFG.EPOCHS}")

    for epoch in range(1, CFG.EPOCHS + 1):
        ep_start = time.time()

        tm = train_one_epoch(model, train_loader, device, criterion,
                             optimizer, scaler, use_amp, CFG.CLIP_GRAD_NORM)
        vm = evaluate(model, val_loader, device, criterion, use_amp, desc="Val")
        scheduler.step()

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_new = optimizer.param_groups[1]["lr"]
        ep_time = time.time() - ep_start

        row = {
            "epoch": epoch,
            "lr_backbone": lr_backbone,
            "lr_new": lr_new,
            "time_sec": ep_time,
            "train_loss": tm["loss"],
            "train_angle_mae_deg": tm["angle_mae_deg"],
            "train_angle_rmse_deg": tm["angle_rmse_deg"],
            "train_grad_norm": tm["grad_norm"],
            "train_mean_logvar": tm["mean_logvar"],
            "train_mean_std_deg": tm["mean_std_deg"],
            "val_loss": vm["loss"],
            "val_angle_mae_deg": vm["angle_mae_deg"],
            "val_angle_rmse_deg": vm["angle_rmse_deg"],
            "val_angle_r2": vm["angle_r2"],
            "val_mean_logvar": vm["mean_logvar"],
            "val_mean_std_deg": vm["mean_std_deg"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{CFG.EPOCHS} | "
            f"TrainMAE={tm['angle_mae_deg']:.3f}° | "
            f"ValMAE={vm['angle_mae_deg']:.3f}° "
            f"RMSE={vm['angle_rmse_deg']:.3f}° | "
            f"Std={vm['mean_std_deg']:.2f}° | "
            f"Time={ep_time:.1f}s"
        )

        if vm["angle_mae_deg"] < best_val_mae:
            best_val_mae = vm["angle_mae_deg"]
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  ✓ Saved best_model.pth (epoch={best_epoch}, val_mae={best_val_mae:.4f}°)")
        else:
            patience += 1
            if patience >= CFG.EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {CFG.EARLY_STOP_PATIENCE} epochs without MAE improvement.")
                break

        write_csv(
            os.path.join(save_dir, "tables", "epoch_metrics.csv"),
            history, list(history[0].keys())
        )

    total_time = time.time() - t0
    print(f"\nTraining done in {total_time / 60:.1f} min | Best epoch={best_epoch}, val_mae={best_val_mae:.4f}°")

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device))
    best_val = evaluate(model, val_loader, device, criterion, use_amp, save_details=True, desc="Best-Val")
    best_test = evaluate(model, test_loader, device, criterion, use_amp, save_details=True, desc="Best-Test")

    print("\n=== Best Model ===")
    print(
        f"[VAL ] MAE={best_val['angle_mae_deg']:.3f}° RMSE={best_val['angle_rmse_deg']:.3f}° "
        f"Uncertainty(avg std)={best_val['mean_std_deg']:.2f}°")
    print(
        f"[TEST] MAE={best_test['angle_mae_deg']:.3f}° RMSE={best_test['angle_rmse_deg']:.3f}° "
        f"Uncertainty(avg std)={best_test['mean_std_deg']:.2f}°")

    fields = ["Split", "Angle_MAE_deg", "Angle_RMSE_deg", "Loss", "Angle_R2", "Mean_LogVar", "Mean_Std_deg"]
    with open(os.path.join(save_dir, "tables", "val_test_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for name, m in [("VAL", best_val), ("TEST", best_test)]:
            w.writerow([
                name,
                m["angle_mae_deg"],
                m["angle_rmse_deg"],
                m["loss"],
                m["angle_r2"],
                m["mean_logvar"],
                m["mean_std_deg"],
            ])

    for tag, m in [("val", best_val), ("test", best_test)]:
        if m["details"]:
            write_csv(
                os.path.join(save_dir, "tables", f"best_{tag}_details.csv"),
                m["details"], list(m["details"][0].keys())
            )

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print("Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DINOv3-ConvNeXt for acute fibre-orientation regression."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Dataset root containing train/val/test CSV and images directories.",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS else None,
        help="Compatible local DINOv3 ConvNeXt-Tiny checkpoint.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path(CFG.RUNS_DIR))
    parser.add_argument("--device", default=CFG.DEVICE or "auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    parser.add_argument("--require-pretrained", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument("--no-uncertainty", action="store_true")
    return parser


def configure_from_args(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.expanduser()
    for split in ("train", "val", "test"):
        setattr(CFG, f"PATCH_CSV_{split.upper()}", str(data_dir / split / f"{split}.csv"))
        setattr(CFG, f"{split.upper()}_ROOT", str(data_dir / split / "images"))
    weight_path = str(args.weights.expanduser()) if args.weights else ""
    CFG.DINOV3_CONVNEXT_PTH = weight_path
    CFG.CONVNEXT_LOCAL_PTH = weight_path
    CFG.RUNS_DIR = str(args.runs_dir.expanduser())
    CFG.DEVICE = "" if args.device == "auto" else args.device
    CFG.FORCE_CUDA = args.device.startswith("cuda")
    CFG.CUDA_DEVICE = args.device if CFG.FORCE_CUDA else "cuda:0"
    CFG.EPOCHS = args.epochs
    CFG.BATCH_SIZE = args.batch_size
    CFG.NUM_WORKERS = args.num_workers
    CFG.SEED = args.seed
    CFG.REQUIRE_PRETRAINED = args.require_pretrained
    CFG.USE_ATTENTION = not args.no_attention
    CFG.USE_FFT = not args.no_fft
    CFG.USE_UNCERTAINTY = not args.no_uncertainty


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    configure_from_args(build_arg_parser().parse_args())
    main()
