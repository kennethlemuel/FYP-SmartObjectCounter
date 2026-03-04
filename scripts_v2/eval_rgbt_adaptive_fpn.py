import os
import sys
import time
import json
import argparse
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.rgbt_cc import RGBTCC_PairedDataset
from models_v2.rgbt_adaptive_fpn import AdaptiveFPNRGBT


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only = True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, strict: bool = False) -> None:
    ckpt = torch.load(ckpt_path, map_location = device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict = strict)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")


def count_parameters_and_size(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    size_mb = total_bytes / (1024.0 * 1024.0)
    return {
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "model_size_mb": float(size_mb),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, measure_timing: bool = True) -> Dict[str, Any]:
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    n = 0
    time_acc = 0.0

    for batch in loader:
        gt_count = batch[-1]
        if torch.is_tensor(gt_count):
            gt_count = gt_count.float().cpu().numpy()
        else:
            gt_count = np.asarray(gt_count, dtype = np.float32)

        if device.type == "cuda" and measure_timing:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        rgb = batch[0].to(device, non_blocking = True)
        t3 = batch[1].to(device, non_blocking = True)
        pred_den = model(rgb, t3)
        pred_den = torch.nan_to_num(pred_den, nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)
        pred_count = pred_den.sum(dim = (1, 2, 3)).detach().float().cpu().numpy()

        if device.type == "cuda" and measure_timing:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        if measure_timing:
            time_acc += (t1 - t0)

        err = pred_count - gt_count
        sum_abs += float(np.abs(err).sum())
        sum_sq += float((err ** 2).sum())
        n += int(gt_count.shape[0])

    mae = sum_abs / max(1, n)
    rmse = (sum_sq / max(1, n)) ** 0.5
    out = {"mae": mae, "rmse": rmse, "num_images": n}
    if measure_timing and n > 0:
        out["eval_ms_per_image"] = float((time_acc * 1000.0) / n)
        out["eval_fps"] = float(n / max(1e-9, time_acc))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type = str, required = True)
    parser.add_argument("--split", type = str, default = "val", choices = ["train", "val", "test"])
    parser.add_argument("--ckpt", type = str, required = True)
    parser.add_argument("--strict_ckpt", action = "store_true")

    parser.add_argument("--img_h", type = int, default = 768)
    parser.add_argument("--img_w", type = int, default = 1024)
    parser.add_argument("--out_stride", type = int, default = 4)
    parser.add_argument("--sigma", type = float, default = 15.0)

    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--num_workers", type = int, default = 0)
    parser.add_argument("--device", type = str, default = "cuda", choices = ["cuda", "cpu"])

    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--deterministic", action = "store_true")
    parser.add_argument("--load_imagenet", action = "store_true")
    parser.add_argument("--out_json", type = str, default = "")

    args = parser.parse_args()

    set_seed(args.seed, deterministic = args.deterministic)
    device = get_device(args.device)

    dataset = RGBTCC_PairedDataset(
        root = args.root,
        split = args.split,
        img_size = (args.img_h, args.img_w),
        out_stride = args.out_stride,
        sigma = args.sigma,
        return_pts = False,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = (device.type == "cuda"),
        drop_last = False,
        generator = generator,
    )

    model = AdaptiveFPNRGBT(load_imagenet = args.load_imagenet)
    model.to(device)
    load_checkpoint(model, args.ckpt, device, strict = args.strict_ckpt)

    metrics: Dict[str, Any] = {}
    metrics.update({
        "mode": "adaptive_fpn",
        "split": args.split,
        "img_h": args.img_h,
        "img_w": args.img_w,
        "out_stride": args.out_stride,
        "sigma": args.sigma,
        "device": str(device),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
    })
    metrics.update(count_parameters_and_size(model))

    eval_metrics = evaluate(model = model, loader = loader, device = device, measure_timing = True)
    metrics.update(eval_metrics)

    print(json.dumps(metrics, indent = 2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent = 2)
        print(f"[OK] Wrote metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
