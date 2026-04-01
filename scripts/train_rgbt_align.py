import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from datasets.rgbt_cc import (
    RGBTCCDset,
    RGBTCC_RGBDset,
    RGBTCC_TDset,
    RGBTCC_RGBTBaseDset,
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
    build_splits_rgbt_cc,
)

from models.csrnet import CSRNet
from models.resnet_cc import ResNetCount
from models.rgbt_base import CSRNetRGBT_Base
from models.rgbt_adaptive_fpn_lite import CSRNetRGBT_AdaptiveFPNLite
from models.rgbt_adaptive_fpn_lite_cal import CSRNetRGBT_AdaptiveFPNLiteCal
from models.rgbt_adaptive_fpn_lite_cal_align import CSRNetRGBT_AdaptiveFPNLiteCalAlign
from models.rgbt_adaptive_fpn_lite_cal_confidence import CSRNetRGBT_AdaptiveFPNLiteCalConfidence
from models.rgbt_adaptive_fpn_lite_cal_misalign import CSRNetRGBT_AdaptiveFPNLiteCalMisalign
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate


# -----------------------------
# Determinism helpers
# -----------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer pytorch versions
        try:
            torch.use_deterministic_algorithms(True, warn_only = True)
        except Exception:
            pass

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _set_requires_grad(module: nn.Module, req: bool) -> None:
    for p in module.parameters():
        p.requires_grad = req


def _warp_thermal_with_shift(thermal: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
    b, _, h, w = thermal.shape
    theta = thermal.new_zeros((b, 2, 3))
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0

    if w > 1:
        theta[:, 0, 2] = 2.0 * shift_px[:, 0] / float(w - 1)
    if h > 1:
        theta[:, 1, 2] = 2.0 * shift_px[:, 1] / float(h - 1)

    grid = F.affine_grid(theta, size = thermal.shape, align_corners = False)
    return F.grid_sample(
        thermal,
        grid,
        mode = "bilinear",
        padding_mode = "zeros",
        align_corners = False,
    )


def apply_synthetic_thermal_shift(
    x4: torch.Tensor,
    *,
    max_shift_px: float,
    p: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a known synthetic shift to the thermal channel only.

    Returns:
    - shifted stacked input
    - supervision target shift to align thermal back to RGB
    - supervision target confidence (1 when aligned, smaller when shifted)
    """
    if max_shift_px <= 0.0 or p <= 0.0:
        b = x4.shape[0]
        zero_shift = x4.new_zeros((b, 2))
        one_conf = x4.new_ones((b,))
        return x4, zero_shift, one_conf

    b = x4.shape[0]
    device = x4.device
    applied_shift = x4.new_empty((b, 2)).uniform_(-max_shift_px, max_shift_px)
    mask = (torch.rand((b, 1), device = device) < p).float()
    applied_shift = applied_shift * mask

    rgb = x4[:, :3, :, :]
    thermal = x4[:, 3:4, :, :]
    thermal_shifted = _warp_thermal_with_shift(thermal, applied_shift)

    target_shift = -applied_shift
    max_norm = max(1e-6, float(max_shift_px) * (2.0 ** 0.5))
    shift_norm = torch.linalg.vector_norm(applied_shift, dim = 1)
    target_conf = (1.0 - shift_norm / max_norm).clamp_(0.0, 1.0)

    return torch.cat([rgb, thermal_shifted], dim = 1), target_shift, target_conf


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model", "net"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt_obj)}")


def _load_model_ckpt(model: nn.Module, ckpt_path: str, *, strict: bool = False, subprefix: Optional[str] = None) -> Tuple[int, int]:
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location = "cpu")
    sd = _extract_state_dict(ckpt)
    sd = {kk.replace("module.", ""): vv for kk, vv in sd.items()}

    if subprefix is not None:
        subprefix = f"{subprefix}."
        sd_sub = {k[len(subprefix):]: v for k, v in sd.items() if k.startswith(subprefix)}
        if len(sd_sub) >= 10:
            sd = sd_sub

    res = model.load_state_dict(sd, strict = strict)
    return len(res.missing_keys), len(res.unexpected_keys)


# -----------------------------
# Metrics (CSRNet-style)
# -----------------------------
@torch.no_grad()
def _count_from_density(den: torch.Tensor) -> torch.Tensor:
    # den: [B,1,H,W]
    return den.sum(dim = (2, 3))


@torch.no_grad()
def _game(count_pred: torch.Tensor, count_gt: torch.Tensor, grid: int) -> torch.Tensor:
    """
    GAME for counts already aggregated per image does not apply.
    This helper is kept for compatibility; in this script GAME is computed
    via density partitioning below.
    """
    return (count_pred - count_gt).abs()


@torch.no_grad()
def _partition_density(den: torch.Tensor, grid: int) -> torch.Tensor:
    """
    den: [B,1,H,W]
    return: [B, grid*grid] counts per cell
    """
    b, c, h, w = den.shape
    if grid <= 1:
        return den.sum(dim = (2, 3)).view(b, 1)

    # Pad so divisible
    gh = int(np.ceil(h / grid) * grid)
    gw = int(np.ceil(w / grid) * grid)
    if gh != h or gw != w:
        den = F.pad(den, (0, gw - w, 0, gh - h))

    _, _, hp, wp = den.shape
    cell_h = hp // grid
    cell_w = wp // grid
    den = den.view(b, 1, grid, cell_h, grid, cell_w)
    den = den.sum(dim = (3, 5))  # sum within each cell
    den = den.view(b, grid * grid)
    return den


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    mae = 0.0
    rmse = 0.0
    game1 = 0.0
    game2 = 0.0
    game3 = 0.0
    n = 0

    for batch in loader:
        if len(batch) == 5:
            # paired rgb-t
            x_rgb, x_t, den_gt, _img_id, _meta = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t = x_t.to(device, non_blocking = True)
            den_gt = den_gt.to(device, non_blocking = True)

            if amp:
                with torch.autocast(device_type = "cuda", dtype = torch.float16):
                    den_pred = model(x_rgb, x_t)
            else:
                den_pred = model(x_rgb, x_t)

        else:
            # single-modal
            x, den_gt, _img_id, _meta = batch
            x = x.to(device, non_blocking = True)
            den_gt = den_gt.to(device, non_blocking = True)

            if amp:
                with torch.autocast(device_type = "cuda", dtype = torch.float16):
                    den_pred = model(x)
            else:
                den_pred = model(x)

        # Eval stability: enforce finite, non-negative densities before summing.
        den_pred = torch.nan_to_num(den_pred.float(), nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)
        den_gt = torch.nan_to_num(den_gt.float(), nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)

        cnt_pred = den_pred.double().sum(dim = (2, 3)).cpu().numpy()
        cnt_gt = den_gt.double().sum(dim = (2, 3)).cpu().numpy()

        err = np.abs(cnt_pred - cnt_gt)
        mae += float(err.sum())
        rmse += float((err ** 2).sum())

        # GAMEk via density partitioning
        g1 = (_partition_density(den_pred, 2) - _partition_density(den_gt, 2)).abs().sum(dim = 1).cpu().numpy()
        g2 = (_partition_density(den_pred, 4) - _partition_density(den_gt, 4)).abs().sum(dim = 1).cpu().numpy()
        g3 = (_partition_density(den_pred, 8) - _partition_density(den_gt, 8)).abs().sum(dim = 1).cpu().numpy()

        game1 += float(g1.sum())
        game2 += float(g2.sum())
        game3 += float(g3.sum())

        n += cnt_gt.shape[0]

    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "game1": float("nan"), "game2": float("nan"), "game3": float("nan")}

    mae /= n
    rmse = (rmse / n) ** 0.5
    game1 /= n
    game2 /= n
    game3 /= n
    return {"mae": mae, "rmse": rmse, "game1": game1, "game2": game2, "game3": game3}


# -----------------------------
# Train
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type = str, required = True)
    ap.add_argument("--out_dir", type = str, required = True)

    ap.add_argument("--mode", type = str, default = "late", choices = ["rgb", "t", "base", "adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign", "early", "late", "adaptive_late"])
    ap.add_argument("--epochs", type = int, default = 100)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--workers", type = int, default = 4)

    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--deterministic", action = "store_true", help = "Enable deterministic mode")
    ap.add_argument("--val_deterministic", action = "store_true", help = "Force deterministic validation (recommended)")

    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--use_onecycle", action = "store_true")
    ap.add_argument("--max_lr", type = float, default = 1e-5)

    # Adaptive-late gate LR (separate param group)
    ap.add_argument("--gate_lr", type = float, default = None)
    ap.add_argument("--max_gate_lr", type = float, default = None)
    ap.add_argument("--init_base_ckpt", type = str, default = "", help = "Optional: path to a pretrained base checkpoint to warm-start adaptive_fpn_lite.")
    ap.add_argument("--init_rgb_ckpt", type = str, default = "", help = "Optional: path to a pretrained RGB baseline checkpoint to warm-start adaptive_late.rgb_net")
    ap.add_argument("--init_t_ckpt", type = str, default = "", help = "Optional: path to a pretrained T baseline checkpoint to warm-start adaptive_late.t_net")
    ap.add_argument("--teacher_base_ckpt", type = str, default = "", help = "Optional: path to a pretrained rgbt_base checkpoint for training-only distillation.")
    ap.add_argument("--lambda_kd_map", type = float, default = 0.0, help = "Weight for teacher density-map distillation loss.")
    ap.add_argument("--lambda_kd_cnt", type = float, default = 0.0, help = "Weight for teacher count distillation loss.")
    ap.add_argument("--kd_warmup_epochs", type = int, default = 0, help = "Linearly ramp KD loss from 0 to full over this many epochs.")

    ap.add_argument("--freeze_backbones_epochs", type = int, default = 0)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--grad_accum", type = int, default = 1)
    ap.add_argument("--clip_grad", type = float, default = 0.0)
    ap.add_argument("--lambda_cnt", type = float, default = 1e-3,
                    help = "Weight for count-level L1 loss on summed density (helps MAE).")

    ap.add_argument("--crop_size", type = int, default = 224)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--down", type = int, default = 8)
    ap.add_argument("--val_fullres", action = "store_true", help = "Validate on full-res images (uses RGBT-CC val/test split).")
    ap.add_argument("--val_img_h", type = int, default = 768)
    ap.add_argument("--val_img_w", type = int, default = 1024)
    ap.add_argument(
        "--align_max_shift_px",
        type = float,
        default = 4.0,
        help = "Maximum absolute thermal pre-alignment shift in pixels for adaptive_fpn_lite_cal_align.",
    )
    ap.add_argument(
        "--synthetic_shift_px",
        type = float,
        default = 4.0,
        help = "Maximum absolute synthetic thermal shift in pixels for misalignment-aware training.",
    )
    ap.add_argument(
        "--synthetic_shift_p",
        type = float,
        default = 0.75,
        help = "Probability of applying a synthetic thermal shift during misalignment-aware training.",
    )
    ap.add_argument(
        "--lambda_shift_sup",
        type = float,
        default = 0.5,
        help = "Weight for synthetic shift supervision loss in misalignment-aware training.",
    )
    ap.add_argument(
        "--lambda_conf_sup",
        type = float,
        default = 0.2,
        help = "Weight for thermal confidence supervision loss in misalignment-aware training.",
    )
    ap.add_argument(
        "--thermal_conf_floor",
        type = float,
        default = 0.25,
        help = "Lower bound for learned thermal confidence in the misalignment-aware model.",
    )

    args = ap.parse_args()

    # If you don’t specify separate gate learning rates, default them to the backbone LR.
    if args.gate_lr is None:
        args.gate_lr = args.lr
    if args.max_gate_lr is None:
        args.max_gate_lr = args.max_lr

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    set_seed(args.seed, deterministic = bool(args.deterministic))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")
    # Build base splits (official train/val split)
    base_train, base_val = build_splits_rgbt_cc(args.data_root)

    # -----------------------------
    # Train: crop-based (fair protocol)
    # -----------------------------
    is_train_det = bool(args.deterministic)

    if args.mode == "rgb":
        ds_train = RGBTCC_RGBDset(
            base_train,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = True,
            deterministic = is_train_det,
        )
        ds_val = RGBTCC_RGBDset(
            base_val,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = False,
            deterministic = True,
        )
    elif args.mode == "t":
        ds_train = RGBTCC_TDset(
            base_train,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = True,
            deterministic = is_train_det,
        )
        ds_val = RGBTCC_TDset(
            base_val,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = False,
            deterministic = True,
        )
    elif args.mode in ["base", "adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign"]:
        ds_train = RGBTCC_RGBTBaseDset(
            base_train,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = True,
            deterministic = is_train_det,
        )
        ds_val = RGBTCC_RGBTBaseDset(
            base_val,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = False,
            deterministic = True,
        )
    else:
        # early / late / adaptive_late train on paired crops
        ds_train = RGBTCCDset(
            base_train,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = True,
            deterministic = is_train_det,
        )
        ds_val = RGBTCCDset(
            base_val,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = False,
            deterministic = True,
        )

    g = torch.Generator()
    g.manual_seed(args.seed)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g,
    )
    # Optional: override validation with full-resolution images for better val/test alignment.
    if args.val_fullres:
        val_split = "val"
        if not (Path(args.data_root) / "val").exists():
            val_split = "test"

        img_size = (int(args.val_img_h), int(args.val_img_w))
        if args.mode == "rgb":
            ds_val = RGBTCC_RGBDataset(
                root = args.data_root,
                split = val_split,
                img_size = img_size,
                out_stride = args.down,
                sigma = args.sigma,
                return_pts = False,
            )
        elif args.mode == "t":
            ds_val = RGBTCC_TDataset(
                root = args.data_root,
                split = val_split,
                img_size = img_size,
                out_stride = args.down,
                sigma = args.sigma,
                return_pts = False,
            )
        elif args.mode in ["base", "adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign"]:
            ds_val = RGBTCC_EarlyFusionDataset(
                root = args.data_root,
                split = val_split,
                img_size = img_size,
                out_stride = args.down,
                sigma = args.sigma,
                return_pts = False,
            )
        else:
            ds_val = RGBTCC_PairedDataset(
                root = args.data_root,
                split = val_split,
                img_size = img_size,
                out_stride = args.down,
                sigma = args.sigma,
                return_pts = False,
            )
        print(f"[init] val_fullres = True (split={val_split}, img={img_size[0]}x{img_size[1]})")

    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g,
    )

    print(f"[init] train = {len(ds_train)}  val = {len(ds_val)}  workers = {args.workers}")
    if args.mode == "adaptive_fpn_lite_cal_misalign":
        print(
            f"[init] misalign-aware training: "
            f"align_max_shift_px={args.align_max_shift_px} "
            f"synthetic_shift_px={args.synthetic_shift_px} "
            f"synthetic_shift_p={args.synthetic_shift_p} "
            f"lambda_shift_sup={args.lambda_shift_sup} "
            f"lambda_conf_sup={args.lambda_conf_sup} "
            f"thermal_conf_floor={args.thermal_conf_floor}"
        )
    elif args.mode == "adaptive_fpn_lite_cal_conf":
        print(f"[init] confidence-aware thermal fusion: thermal_conf_floor={args.thermal_conf_floor}")

    # Model selection
    if args.mode == "rgb":
        model = ResNetCount(load_imagenet = True).to(device)
    elif args.mode == "t":
        model = ResNetCount(load_imagenet = True).to(device)
    elif args.mode == "base":
        model = CSRNetRGBT_Base(load_imagenet = True).to(device)
    elif args.mode == "adaptive_fpn_lite":
        model = CSRNetRGBT_AdaptiveFPNLite(load_imagenet = True).to(device)

        if args.init_base_ckpt:
            missing, unexpected = _load_model_ckpt(model, args.init_base_ckpt, strict = False)
            print(f"[init] warm-start base from {os.path.expanduser(args.init_base_ckpt)} (missing={missing} unexpected={unexpected})")
    elif args.mode == "adaptive_fpn_lite_cal":
        model = CSRNetRGBT_AdaptiveFPNLiteCal(load_imagenet = True).to(device)

        if args.init_base_ckpt:
            missing, unexpected = _load_model_ckpt(model, args.init_base_ckpt, strict = False)
            print(f"[init] warm-start base from {os.path.expanduser(args.init_base_ckpt)} (missing={missing} unexpected={unexpected})")
    elif args.mode == "adaptive_fpn_lite_cal_align":
        model = CSRNetRGBT_AdaptiveFPNLiteCalAlign(
            load_imagenet = True,
            max_shift_px = args.align_max_shift_px,
        ).to(device)

        if args.init_base_ckpt:
            missing, unexpected = _load_model_ckpt(model, args.init_base_ckpt, strict = False)
            print(f"[init] warm-start base from {os.path.expanduser(args.init_base_ckpt)} (missing={missing} unexpected={unexpected})")
    elif args.mode == "adaptive_fpn_lite_cal_conf":
        model = CSRNetRGBT_AdaptiveFPNLiteCalConfidence(
            load_imagenet = True,
            thermal_conf_floor = args.thermal_conf_floor,
        ).to(device)

        if args.init_base_ckpt:
            missing, unexpected = _load_model_ckpt(model, args.init_base_ckpt, strict = False)
            print(f"[init] warm-start base from {os.path.expanduser(args.init_base_ckpt)} (missing={missing} unexpected={unexpected})")
    elif args.mode == "adaptive_fpn_lite_cal_misalign":
        model = CSRNetRGBT_AdaptiveFPNLiteCalMisalign(
            load_imagenet = True,
            max_shift_px = args.align_max_shift_px,
            thermal_conf_floor = args.thermal_conf_floor,
        ).to(device)

        if args.init_base_ckpt:
            missing, unexpected = _load_model_ckpt(model, args.init_base_ckpt, strict = False)
            print(f"[init] warm-start base from {os.path.expanduser(args.init_base_ckpt)} (missing={missing} unexpected={unexpected})")
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = True).to(device)
    elif args.mode == "late":
        model = CSRNetRGBT_Late(load_imagenet = True).to(device)
    elif args.mode == "adaptive_late":
        # adaptive_late model APIs may differ across versions; support both keyword names.
        try:
            model = CSRNetRGBT_AdaptiveLate(load_imagenet = True).to(device)
        except TypeError:
            model = CSRNetRGBT_AdaptiveLate(load_weights = True).to(device)

        # Optional: warm-start the RGB/T experts from already-trained baseline checkpoints.
        # This is OFF by default (empty path). If you enable it, report it as an ablation/training trick.
        def _load_into(net: nn.Module, ckpt_path: str) -> None:
            if not ckpt_path:
                return
            missing, unexpected = _load_model_ckpt(net, ckpt_path, strict = False, subprefix = "rgb_net")
            if missing > 0 and unexpected > 0:
                missing, unexpected = _load_model_ckpt(net, ckpt_path, strict = False, subprefix = "t_net")
            print(f"[init] warm-start from {os.path.expanduser(ckpt_path)} (missing={missing} unexpected={unexpected})")

        _load_into(model.rgb_net, args.init_rgb_ckpt)
        _load_into(model.t_net, args.init_t_ckpt)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    teacher_model: Optional[nn.Module] = None
    use_kd = bool(args.teacher_base_ckpt) and (args.lambda_kd_map > 0.0 or args.lambda_kd_cnt > 0.0)
    if use_kd:
        if args.mode not in ("base", "adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign"):
            raise ValueError("teacher_base_ckpt distillation is only supported for base-style 4-channel modes.")
        teacher_model = CSRNetRGBT_Base(load_imagenet = True).to(device)
        missing, unexpected = _load_model_ckpt(teacher_model, args.teacher_base_ckpt, strict = False)
        teacher_model.eval()
        _set_requires_grad(teacher_model, False)
        print(
            f"[init] teacher base from {os.path.expanduser(args.teacher_base_ckpt)} "
            f"(missing={missing} unexpected={unexpected}) "
            f"lambda_kd_map={args.lambda_kd_map} lambda_kd_cnt={args.lambda_kd_cnt} "
            f"kd_warmup_epochs={args.kd_warmup_epochs}"
        )

    # Loss
    # Keep MSE over density maps; GT density is count-preserving (sum == number of points).
    def loss_fn(pred: torch.Tensor, gt: torch.Tensor, lambda_cnt: float = 1e-3) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan = 0.0, posinf = 0.0, neginf = 0.0)
        gt = torch.nan_to_num(gt, nan = 0.0, posinf = 0.0, neginf = 0.0)
        den_loss = F.mse_loss(pred, gt, reduction = "sum") / pred.shape[0]

        if lambda_cnt > 0.0:
            pred_cnt = pred.sum(dim = (-2, -1))
            gt_cnt = gt.sum(dim = (-2, -1))
            cnt_loss = F.l1_loss(pred_cnt, gt_cnt, reduction = "mean")
            return den_loss + lambda_cnt * cnt_loss

        return den_loss

    def kd_loss_fn(
        pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        lambda_kd_map: float,
        lambda_kd_cnt: float,
    ) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)
        teacher_pred = torch.nan_to_num(teacher_pred, nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)

        loss = pred.new_zeros(())
        if lambda_kd_map > 0.0:
            map_loss = F.mse_loss(pred, teacher_pred, reduction = "sum") / pred.shape[0]
            loss = loss + lambda_kd_map * map_loss
        if lambda_kd_cnt > 0.0:
            pred_cnt = pred.sum(dim = (-2, -1))
            teacher_cnt = teacher_pred.sum(dim = (-2, -1))
            cnt_loss = F.l1_loss(pred_cnt, teacher_cnt, reduction = "mean")
            loss = loss + lambda_kd_cnt * cnt_loss
        return loss

    def misalign_aux_loss_fn(
        pred_shift_px: torch.Tensor,
        target_shift_px: torch.Tensor,
        thermal_conf: torch.Tensor,
        target_conf: torch.Tensor,
    ) -> torch.Tensor:
        shift_scale = max(1e-6, float(args.align_max_shift_px))
        pred_shift_norm = pred_shift_px / shift_scale
        target_shift_norm = target_shift_px / shift_scale
        shift_loss = F.smooth_l1_loss(pred_shift_norm, target_shift_norm, reduction = "mean")
        conf_loss = F.mse_loss(thermal_conf, target_conf, reduction = "mean")
        return args.lambda_shift_sup * shift_loss + args.lambda_conf_sup * conf_loss

    
    scaler = GradScaler(enabled = bool(args.amp))

    # Optimizer and LR schedule
    if args.mode == "adaptive_late":
        backbone_params = list(model.rgb_net.parameters()) + list(model.t_net.parameters())
        gate_module = getattr(model, "gate", None)
        if gate_module is None:
            gate_module = getattr(model, "gate_net", None)
        if gate_module is None:
            raise AttributeError("CSRNetRGBT_AdaptiveLate must expose a gate module as .gate or .gate_net")
        gate_params = list(gate_module.parameters())
        # Calibration layers are part of adaptive fusion and must be optimized.
        cal_params = []
        for cal_name in ("rgb_cal", "t_cal"):
            cal_mod = getattr(model, cal_name, None)
            if cal_mod is not None:
                cal_params.extend(list(cal_mod.parameters()))
        adaptive_head_params = gate_params + cal_params
        params = [
            {"params": backbone_params, "lr": args.lr},
            {"params": adaptive_head_params, "lr": args.gate_lr},
        ]
        print(
            f"[init] adaptive optimizer groups: "
            f"backbone_params={sum(p.numel() for p in backbone_params)} "
            f"head_params={sum(p.numel() for p in adaptive_head_params)}"
        )
    elif args.mode in ("adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign"):
        base_module_names = ("frontend", "backend", "output_layer")
        head_module_names = (
            "lat2", "lat3", "lat4", "latb", "scale_gate", "refine", "residual_out", "count_head",
            "align_feat", "align_fc", "shift_head", "conf_head",
        )

        backbone_params = []
        adaptive_head_params = []

        for name in base_module_names:
            mod = getattr(model, name, None)
            if mod is not None:
                backbone_params.extend(list(mod.parameters()))

        for name in head_module_names:
            mod = getattr(model, name, None)
            if mod is not None:
                adaptive_head_params.extend(list(mod.parameters()))
        if hasattr(model, "residual_scale"):
            adaptive_head_params.append(model.residual_scale)

        params = [
            {"params": backbone_params, "lr": args.lr},
            {"params": adaptive_head_params, "lr": args.gate_lr},
        ]
        print(
            f"[init] adaptive_lite optimizer groups: "
            f"backbone_params={sum(p.numel() for p in backbone_params)} "
            f"head_params={sum(p.numel() for p in adaptive_head_params)}"
        )
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr = args.lr, weight_decay = args.weight_decay)

    if args.use_onecycle:
        if args.mode in ("adaptive_late", "adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign"):
            max_lrs = [args.max_lr, args.max_gate_lr]
        else:
            max_lrs = args.max_lr

        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr = max_lrs,
            total_steps = args.epochs * max(1, len(dl_train)),
            pct_start = 0.1,
            div_factor = 10.0,
            final_div_factor = 100.0,
            anneal_strategy = "cos",
        )
    else:
        sch = None

    best_mae = float("inf")
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    # -----------------------------
    # Train loop
    # -----------------------------
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if teacher_model is not None:
            teacher_model.eval()

        if args.kd_warmup_epochs > 0:
            kd_scale = min(1.0, float(ep) / float(args.kd_warmup_epochs))
        else:
            kd_scale = 1.0

        if args.mode == "adaptive_late" and args.freeze_backbones_epochs > 0:
            # freeze_experts = (ep <= args.freeze_backbones_epochs)  # kept for readability
            _set_requires_grad(model.rgb_net, ep > args.freeze_backbones_epochs)
            _set_requires_grad(model.t_net, ep > args.freeze_backbones_epochs)

            gate_module = getattr(model, "gate", None)
            if gate_module is None:
                gate_module = getattr(model, "gate_net", None)
            if gate_module is None:
                raise AttributeError("CSRNetRGBT_AdaptiveLate must expose a gate module as .gate or .gate_net")
            # Freeze/unfreeze experts using freeze_backbones_epochs, but keep gate trainable.
            _set_requires_grad(gate_module, True)
        elif args.mode in ("adaptive_fpn_lite", "adaptive_fpn_lite_cal", "adaptive_fpn_lite_cal_align", "adaptive_fpn_lite_cal_conf", "adaptive_fpn_lite_cal_misalign") and args.freeze_backbones_epochs > 0:
            for name in ("frontend", "backend", "output_layer"):
                mod = getattr(model, name, None)
                if mod is not None:
                    _set_requires_grad(mod, ep > args.freeze_backbones_epochs)
            for name in ("lat2", "lat3", "lat4", "latb", "scale_gate", "refine", "residual_out", "count_head", "align_feat", "align_fc", "shift_head", "conf_head"):
                mod = getattr(model, name, None)
                if mod is not None:
                    _set_requires_grad(mod, True)

        running = 0.0
        step = 0

        opt.zero_grad(set_to_none = True)

        for batch in dl_train:
            step += 1

            if args.mode in ["late", "adaptive_late"]:
                x_rgb, x_t, den_gt, _img_id, _meta = batch
                x_rgb = x_rgb.to(device, non_blocking = True)
                x_t = x_t.to(device, non_blocking = True)
                den_gt = den_gt.to(device, non_blocking = True)

                if args.amp:
                    with torch.autocast(device_type = "cuda", dtype = torch.float16):
                        den_pred = model(x_rgb, x_t)
                        loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)
                else:
                    den_pred = model(x_rgb, x_t)
                    loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)

            elif args.mode == "early":
                x_rgb, x_t, den_gt, _img_id, _meta = batch
                # For early-fusion model, build 4ch in dataset already as x_rgb (RGB) and x_t (T) are 3ch,
                # but our early dataset in this project returns (rgb, t, den) via RGBTCCDset, so we keep CSRNetRGBT_Early
                # expecting (rgb, t). It internally concatenates if needed.
                x_rgb = x_rgb.to(device, non_blocking = True)
                x_t = x_t.to(device, non_blocking = True)
                den_gt = den_gt.to(device, non_blocking = True)

                if args.amp:
                    with torch.autocast(device_type = "cuda", dtype = torch.float16):
                        den_pred = model(x_rgb, x_t)
                        loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)
                else:
                    den_pred = model(x_rgb, x_t)
                    loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)

            else:
                x, den_gt, _img_id, _meta = batch
                x = x.to(device, non_blocking = True)
                den_gt = den_gt.to(device, non_blocking = True)
                target_shift_px = None
                target_conf = None

                if args.mode == "adaptive_fpn_lite_cal_misalign":
                    x, target_shift_px, target_conf = apply_synthetic_thermal_shift(
                        x,
                        max_shift_px = float(args.synthetic_shift_px),
                        p = float(args.synthetic_shift_p),
                    )

                if args.amp:
                    with torch.autocast(device_type = "cuda", dtype = torch.float16):
                        if args.mode == "adaptive_fpn_lite_cal_misalign":
                            den_pred, aux = model(x, return_aux = True)
                        else:
                            den_pred = model(x)
                        loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)
                        if args.mode == "adaptive_fpn_lite_cal_misalign":
                            loss = loss + misalign_aux_loss_fn(
                                aux["pred_shift_px"],
                                target_shift_px,
                                aux["thermal_conf"],
                                target_conf,
                            )
                        if teacher_model is not None:
                            with torch.no_grad():
                                teacher_pred = teacher_model(x)
                            loss = loss + kd_scale * kd_loss_fn(
                                den_pred,
                                teacher_pred,
                                lambda_kd_map = args.lambda_kd_map,
                                lambda_kd_cnt = args.lambda_kd_cnt,
                            )
                else:
                    if args.mode == "adaptive_fpn_lite_cal_misalign":
                        den_pred, aux = model(x, return_aux = True)
                    else:
                        den_pred = model(x)
                    loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)
                    if args.mode == "adaptive_fpn_lite_cal_misalign":
                        loss = loss + misalign_aux_loss_fn(
                            aux["pred_shift_px"],
                            target_shift_px,
                            aux["thermal_conf"],
                            target_conf,
                        )
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_pred = teacher_model(x)
                        loss = loss + kd_scale * kd_loss_fn(
                            den_pred,
                            teacher_pred,
                            lambda_kd_map = args.lambda_kd_map,
                            lambda_kd_cnt = args.lambda_kd_cnt,
                        )

            loss = loss / max(1, int(args.grad_accum))

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % max(1, int(args.grad_accum))) == 0:
                if args.clip_grad and args.clip_grad > 0:
                    if args.amp:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                if args.amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none = True)

                if sch is not None:
                    sch.step()

            running += float(loss.detach().item()) * max(1, int(args.grad_accum))

            if step == 1 or (step % 50) == 0:
                print(f"[e{ep:03d} s{step:04d}/{len(dl_train)}] loss = {running / step:.6f}")

        train_loss = running / max(1, step)

        # Validation
        metrics = eval_one_epoch(model, dl_val, device, amp = bool(args.amp))
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        g1 = metrics["game1"]
        g2 = metrics["game2"]
        g3 = metrics["game3"]

        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"Epoch {ep:03d}: train_loss = {train_loss:.6f}  "
            f"MAE/GAME0 = {mae:.3f}  RMSE = {rmse:.3f}  "
            f"GAME1 = {g1:.3f}  GAME2 = {g2:.3f}  GAME3 = {g3:.3f}  "
            f"lr = {lr_now:.2e}  kd = {kd_scale:.2f}  time = {dt:.1f}s"
        )

        # Save last
        ckpt = {
            "epoch": ep,
            "args": vars(args),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_mae": best_mae,
        }
        torch.save(ckpt, last_path)

        # Save best
        if mae < best_mae:
            best_mae = mae
            torch.save(ckpt, best_path)

    print(f"[done] best_mae = {best_mae:.3f}  best_path = {best_path}")


if __name__ == "__main__":
    main()
