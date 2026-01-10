import os
import sys
import math
import argparse
import random
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.csrnet import CSRNet
from models.rgbt_late import CSRNetRGBT_Late
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate
from datasets.rgbt_cc import (
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_density_sum_preserving(den: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    old_h, old_w = den.shape[-2], den.shape[-1]
    new_h, new_w = int(size_hw[0]), int(size_hw[1])

    if old_h == new_h and old_w == new_w:
        return den

    den_rs = F.interpolate(den, size = (new_h, new_w), mode = "bilinear", align_corners = False)
    den_rs = den_rs * (old_h * old_w) / float(new_h * new_w)
    return den_rs


def game_error(pred: torch.Tensor, gt: torch.Tensor, level: int) -> float:
    b, _, h, w = pred.shape
    k = 2 ** int(level)

    gh = max(1, h // k)
    gw = max(1, w // k)
    h2 = gh * k
    w2 = gw * k

    pred_c = pred[:, :, :h2, :w2]
    gt_c = gt[:, :, :h2, :w2]

    pred_cells = pred_c.view(b, 1, k, gh, k, gw).sum(dim = (3, 5))
    gt_cells = gt_c.view(b, 1, k, gh, k, gw).sum(dim = (3, 5))

    err = torch.abs(pred_cells - gt_cells).sum(dim = (2, 3))
    return float(err.mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mode: str, game_levels = (0, 1, 2, 3)):
    model.eval()

    rmse_acc = 0.0
    mae_acc = 0.0
    game_acc = {L: 0.0 for L in game_levels}

    n = 0
    for batch in loader:
        if mode in ["rgb", "t"]:
            x, den, _name, _gtc = batch
            x = x.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x)

        elif mode == "early":
            x4, den, _name, _gtc = batch
            x4 = x4.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x4)

        else:
            x_rgb, x_t3, den, _name, _gtc = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x_rgb, x_t3)

        if pred.shape[-2:] != den.shape[-2:]:
            den = resize_density_sum_preserving(den, pred.shape[-2:])

        c_pred = float(pred.sum().item())
        c_gt = float(den.sum().item())
        err = (c_pred - c_gt)

        mae_acc += abs(err)
        rmse_acc += (err ** 2)

        for L in game_levels:
            game_acc[L] += game_error(pred, den, level = L)

        n += 1

    n = max(1, n)
    out = {
        "MAE": mae_acc / n,
        "RMSE": math.sqrt(rmse_acc / n),
    }
    for L in game_levels:
        out[f"GAME{L}"] = game_acc[L] / n
    return out


def build_model(mode: str, load_imagenet: bool = False) -> nn.Module:
    if mode == "rgb":
        return CSRNet(load_imagenet = load_imagenet)
    if mode == "t":
        return CSRNet(load_imagenet = load_imagenet)
    if mode == "early":
        from models.rgbt_early import CSRNetRGBT_Early
        return CSRNetRGBT_Early(load_imagenet = load_imagenet)
    if mode == "late":
        return CSRNetRGBT_Late(load_imagenet = load_imagenet)
    if mode == "adaptive_late":
        return CSRNetRGBT_AdaptiveLate(load_imagenet = load_imagenet)
    raise ValueError(f"Unknown mode: {mode}")


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if all(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _extract_state_dict(ck: Any, use_ema: bool) -> Dict[str, torch.Tensor]:
    if isinstance(ck, dict):
        if use_ema:
            for key in ["ema_model", "model_ema", "ema", "ema_state_dict"]:
                if key in ck and isinstance(ck[key], dict):
                    return ck[key]
        for key in ["model", "state_dict", "net", "model_state_dict"]:
            if key in ck and isinstance(ck[key], dict):
                return ck[key]

    if isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck

    raise ValueError("Checkpoint format not recognized (cannot find model state_dict).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["rgb", "t", "early", "late", "adaptive_late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split", choices = ["train", "val", "test"], required = True)

    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--out_stride", type = int, default = 8)

    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--use_ema", action = "store_true")
    ap.add_argument("--num_workers", type = int, default = 4)

    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--deterministic", action = "store_true")

    args = ap.parse_args()

    device = get_device()
    print(f"[init] device = {device}")

    set_seed(args.seed, deterministic = args.deterministic)

    img_size = (args.img_h, args.img_w)

    if args.mode == "rgb":
        ds = RGBTCC_RGBDataset(args.data_root, args.split, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)
    elif args.mode == "t":
        ds = RGBTCC_TDataset(args.data_root, args.split, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)
    elif args.mode == "early":
        ds = RGBTCC_EarlyFusionDataset(args.data_root, args.split, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)
    else:
        ds = RGBTCC_PairedDataset(args.data_root, args.split, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)

    loader = DataLoader(
        ds,
        batch_size = 1,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = False,
    )

    ck = torch.load(args.ckpt, map_location = "cpu")

    if isinstance(ck, dict):
        epoch = ck.get("epoch", None)
        best_rmse = ck.get("best_rmse", None)
        if epoch is not None:
            msg = f"[ckpt] epoch = {epoch}"
            if best_rmse is not None:
                msg += f" | best_rmse = {best_rmse}"
            print(msg)

    state = _extract_state_dict(ck, use_ema = args.use_ema)
    state = _strip_module_prefix(state)

    model = build_model(args.mode, load_imagenet = False).to(device)
    model.load_state_dict(state, strict = True)

    if args.use_ema:
        print("[eval] using EMA weights from checkpoint")
    else:
        print("[eval] using raw model weights from checkpoint")

    stats = evaluate(model, loader, device, mode = args.mode)
    print(
        f"[split {args.split}] "
        f"MAE = {stats['MAE']:.3f}, RMSE = {stats['RMSE']:.3f}, "
        f"GAME0 = {stats['GAME0']:.3f}, GAME1 = {stats['GAME1']:.3f}, "
        f"GAME2 = {stats['GAME2']:.3f}, GAME3 = {stats['GAME3']:.3f}"
    )


if __name__ == "__main__":
    main()
