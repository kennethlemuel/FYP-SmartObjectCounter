import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure scripts/ is importable (so `import train_rgbt` works reliably)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_CANDIDATES = [
    _THIS_DIR,
    os.path.dirname(_THIS_DIR),
    os.path.dirname(os.path.dirname(_THIS_DIR)),
]
PROJECT_ROOT = None
for c in _CANDIDATES:
    if os.path.isdir(os.path.join(c, "models")) and os.path.isdir(os.path.join(c, "datasets")):
        PROJECT_ROOT = c
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.path.dirname(_THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# <-- This was missing and caused your NameError
import train_rgbt

from datasets.rgbt_cc import (
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
)


def _safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location = map_location, weights_only = True)
    except TypeError:
        return torch.load(path, map_location = map_location)


def _build_dataset(mode: str, data_root: str, split: str, img_h: int, img_w: int, sigma: float, out_stride: int):
    img_size = (int(img_h), int(img_w))

    if mode == "rgb":
        return RGBTCC_RGBDataset(
            data_root, split, img_size, sigma,
            return_pts = False, out_stride = out_stride
        )

    if mode == "t":
        return RGBTCC_TDataset(
            data_root, split, img_size, sigma,
            return_pts = False, out_stride = out_stride
        )

    if mode == "early":
        return RGBTCC_EarlyFusionDataset(
            data_root, split, img_size, sigma,
            return_pts = False, out_stride = out_stride
        )

    if mode in ["late", "adaptive_late"]:
        return RGBTCC_PairedDataset(
            data_root, split, img_size, sigma,
            return_pts = False, out_stride = out_stride
        )

    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["rgb", "t", "early", "late", "adaptive_late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split", choices = ["val", "test", "train"], default = "test")

    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--out_stride", type = int, default = 8)

    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--use_ema", action = "store_true")
    ap.add_argument("--num_workers", type = int, default = 4)
    ap.add_argument("--seed", type = int, default = 42)

    args = ap.parse_args()

    assert args.img_h % args.out_stride == 0, "img_h must be divisible by out_stride"
    assert args.img_w % args.out_stride == 0, "img_w must be divisible by out_stride"

    train_rgbt.set_seed(args.seed, deterministic = True)
    device = train_rgbt.get_device()
    print(f"[init] device = {device}")
    print(f"[init] PROJECT_ROOT = {PROJECT_ROOT}")

    ds = _build_dataset(
        mode = args.mode,
        data_root = args.data_root,
        split = args.split,
        img_h = args.img_h,
        img_w = args.img_w,
        sigma = args.sigma,
        out_stride = args.out_stride,
    )

    loader = DataLoader(
        ds,
        batch_size = 1,
        shuffle = False,
        num_workers = int(args.num_workers),
        pin_memory = True,
        drop_last = False,
    )

    model = train_rgbt.build_model(args.mode, load_imagenet = True).to(device)

    ckpt_obj = _safe_torch_load(args.ckpt, map_location = "cpu")

    if isinstance(ckpt_obj, dict) and ("model" in ckpt_obj or "ema_model" in ckpt_obj):
        if args.use_ema and ("ema_model" in ckpt_obj) and (ckpt_obj["ema_model"] is not None):
            sd = ckpt_obj["ema_model"]
            print("[eval] using EMA weights from checkpoint")
        else:
            sd = ckpt_obj["model"]
            print("[eval] using raw model weights from checkpoint")

        model.load_state_dict(sd, strict = True)
        if "epoch" in ckpt_obj:
            print(f"[ckpt] epoch = {ckpt_obj['epoch']}")
    else:
        print("[ckpt] loading raw state_dict")
        model.load_state_dict(ckpt_obj, strict = True)

    stats = train_rgbt.evaluate(model, loader, device, mode = args.mode)
    print(f"[split {args.split}] MAE = {stats['MAE']:.3f}, RMSE = {stats['RMSE']:.3f}, "
          f"GAME0 = {stats['GAME0']:.3f}, GAME1 = {stats['GAME1']:.3f}, GAME2 = {stats['GAME2']:.3f}, GAME3 = {stats['GAME3']:.3f}")


if __name__ == "__main__":
    main()
