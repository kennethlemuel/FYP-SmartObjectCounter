import os
import sys
import time
import json
import math
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

# Dataset loaders (protocol-fixed)
from datasets.rgbt_cc import (
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
)

#Models
from models.csrnet import CSRNet
from models.resnet_cc import ResNetCount
from models.rgbt_base import CSRNetRGBT_Base
from models.rgbt_adaptive_fpn_lite import CSRNetRGBT_AdaptiveFPNLite
try:
    from models.rgbt_early import CSRNetRGBT_EarlyFusion as CSRNetRGBT_Early
except ImportError:
    from models.rgbt_early import CSRNetRGBT_Early

try:
    from models.rgbt_late import CSRNetRGBT_LateFusion as CSRNetRGBT_Late
except ImportError:
    from models.rgbt_late import CSRNetRGBT_Late

try:
    from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLateFusion as CSRNetRGBT_AdaptiveLate
except ImportError:
    from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate


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

def remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Best-effort key remaps for backward compatibility across refactors."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        kk = k
        # Older adaptive-late checkpoints sometimes used "gate_net.*" naming.
        if kk.startswith("gate_net."):
            kk = "gate." + kk[len("gate_net."):]
        out[kk] = v
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
    sd = remap_checkpoint_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict = strict)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")


def build_model(mode: str, load_imagenet: bool) -> torch.nn.Module:
    mode = mode.lower()
    if mode == "rgb":
        return ResNetCount(load_imagenet = load_imagenet)
    if mode == "t":
        #Thermal is provided as 3-channel tensor (replicated) by the dataset loader.
        return ResNetCount(load_imagenet = load_imagenet)
    if mode == "base":
        return CSRNetRGBT_Base(load_imagenet = load_imagenet)
    if mode == "adaptive_fpn_lite":
        return CSRNetRGBT_AdaptiveFPNLite(load_imagenet = load_imagenet)
    if mode == "early":
        return CSRNetRGBT_Early(load_imagenet = load_imagenet)
    if mode == "late":
        return CSRNetRGBT_Late(load_imagenet = load_imagenet)
    if mode == "adaptive_late":
        return CSRNetRGBT_AdaptiveLate(load_imagenet = load_imagenet)

    raise ValueError(f"Unknown mode: {mode}")


def build_dataset(
    mode: str,
    root: str,
    split: str,
    img_h: int,
    img_w: int,
    out_stride: int,
    sigma: float,
) -> torch.utils.data.Dataset:
    img_size = (img_h, img_w)
    mode = mode.lower()

    if mode == "rgb":
        return RGBTCC_RGBDataset(
            root = root,
            split = split,
            img_size = img_size,
            out_stride = out_stride,
            sigma = sigma,
            return_pts = False,
        )
    if mode == "t":
        return RGBTCC_TDataset(
            root = root,
            split = split,
            img_size = img_size,
            out_stride = out_stride,
            sigma = sigma,
            return_pts = False,
        )
    if mode in ["base", "adaptive_fpn_lite"]:
        return RGBTCC_EarlyFusionDataset(
            root = root,
            split = split,
            img_size = img_size,
            out_stride = out_stride,
            sigma = sigma,
            return_pts = False,
        )
    if mode == "early":
        # Evaluate early-fusion using the paired RGB + thermal loader to match the two-input model API.
        return RGBTCC_PairedDataset(
            root = root,
            split = split,
            img_size = img_size,
            out_stride = out_stride,
            sigma = sigma,
            return_pts = False,
        )

    # late/adaptive_late both need paired rgb + thermal
    return RGBTCC_PairedDataset(
        root = root,
        split = split,
        img_size = img_size,
        out_stride = out_stride,
        sigma = sigma,
        return_pts = False,
    )

@torch.no_grad()
def forward_density(model: torch.nn.Module, mode: str, batch: Tuple[Any, ...], device: torch.device) -> torch.Tensor:
    """
    Returns predicted density map tensor on device with shape [B, 1, H', W'].
    Handles different model input conventions and tuple outputs.
    """
    mode = mode.lower()

    if mode in ["rgb", "t", "base", "adaptive_fpn_lite"]:
        #dataset returns: (x, den, fname, gt_count)
        x = batch[0].to(device, non_blocking = True)
        out = model(x)
    else:
        #paired returns: (rgb, t3, den, fname, gt_count)
        rgb = batch[0].to(device, non_blocking = True)
        t3 = batch[1].to(device, non_blocking = True)
        out = model(rgb, t3)

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError("Model forward did not return a tensor density map.")

    return out


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
def benchmark_forward(
    model: torch.nn.Module,
    mode: str,
    batch: Tuple[Any, ...],
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    """
    Micro-benchmark on a single batch (repeat forward pass).
    Returns avg latency (ms) and peak GPU memory (MB), if CUDA is available.
    """
    model.eval()

    #Move inputs to device once, avoid data transfer in timing loop.
    if mode.lower() in ["rgb", "t", "base"]:
        x = batch[0].to(device, non_blocking = True)
        batch_on_device = (x,)
    else:
        rgb = batch[0].to(device, non_blocking = True)
        t3 = batch[1].to(device, non_blocking = True)
        batch_on_device = (rgb, t3)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(max(0, warmup)):
        if mode.lower() in ["rgb", "t", "base"]:
            out = model(batch_on_device[0])
        else:
            out = model(batch_on_device[0], batch_on_device[1])
        if isinstance(out, (tuple, list)):
            out = out[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        if mode.lower() in ["rgb", "t", "base"]:
            out = model(batch_on_device[0])
        else:
            out = model(batch_on_device[0], batch_on_device[1])
        if isinstance(out, (tuple, list)):
            out = out[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / max(1, iters)

    result: Dict[str, Any] = {"bench_forward_ms": float(avg_ms)}
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        result["peak_mem_alloc_mb"] = float(peak_alloc)
        result["peak_mem_reserved_mb"] = float(peak_reserved)

    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    mode: str,
    loader: DataLoader,
    device: torch.device,
    measure_timing: bool = True,
) -> Dict[str, Any]:
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

        pred_den = forward_density(model, mode, batch, device)
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
        n += int(err.shape[0])

    mae = sum_abs / max(1, n)
    rmse = math.sqrt(sum_sq / max(1, n))

    out: Dict[str, Any] = {"mae": float(mae), "rmse": float(rmse), "num_images": int(n)}
    if measure_timing and n > 0:
        out["eval_ms_per_image"] = float((time_acc * 1000.0) / n)
        out["eval_fps"] = float(n / max(1e-9, time_acc))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type = str, required = True, help = "Dataset root (contains train/val/test folders).")
    parser.add_argument("--split", type = str, default = "val", choices = ["train", "val", "test"])
    parser.add_argument("--mode", type = str, required = True, choices = ["rgb", "t", "base", "adaptive_fpn_lite", "early", "late", "adaptive_late"])
    parser.add_argument("--ckpt", type = str, required = True, help = "Path to checkpoint (.pth).")
    parser.add_argument("--strict_ckpt", action = "store_true", help = "Load checkpoint with strict=True (fail on missing/unexpected keys).")
    parser.add_argument("--zero_cal_bias", action = "store_true", help = "After loading, zero rgb_cal/t_cal biases (debug constant-offset blow-up).")

    #Fair default for SOTA comparisons: evaluate on full resolution used by RGBT-CC (768x1024).
    parser.add_argument("--img_h", type = int, default = 768)
    parser.add_argument("--img_w", type = int, default = 1024)

    parser.add_argument("--out_stride", type = int, default = 8)
    parser.add_argument("--sigma", type = float, default = 15.0)

    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--num_workers", type = int, default = 0, help = "Use 0 for strict determinism.")
    parser.add_argument("--device", type = str, default = "cuda", choices = ["cuda", "cpu"])

    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--deterministic", action = "store_true")

    parser.add_argument("--load_imagenet", action = "store_true")

    #Benchmarking
    parser.add_argument("--benchmark", action = "store_true", help = "Run forward-pass benchmark on 1 batch.")
    parser.add_argument("--bench_warmup", type = int, default = 10)
    parser.add_argument("--bench_iters", type = int, default = 100)

    parser.add_argument("--out_json", type = str, default = "")

    args = parser.parse_args()

    set_seed(args.seed, deterministic = args.deterministic)
    device = get_device(args.device)

    dataset = build_dataset(
        mode = args.mode,
        root = args.root,
        split = args.split,
        img_h = args.img_h,
        img_w = args.img_w,
        out_stride = args.out_stride,
        sigma = args.sigma,
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

    model = build_model(mode = args.mode, load_imagenet = args.load_imagenet)
    model.to(device)
    load_checkpoint(model, args.ckpt, device, strict = args.strict_ckpt)
    if args.zero_cal_bias:
        for name in ["rgb_cal", "t_cal"]:
            layer = getattr(model, name, None)
            if layer is not None and getattr(layer, "bias", None) is not None:
                with torch.no_grad():
                    layer.bias.zero_()

    metrics: Dict[str, Any] = {}
    metrics.update({
        "mode": args.mode,
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

    if args.benchmark:
        first_batch = next(iter(loader))
        bench = benchmark_forward(
            model = model,
            mode = args.mode,
            batch = first_batch,
            device = device,
            warmup = args.bench_warmup,
            iters = args.bench_iters,
        )
        metrics.update(bench)

    eval_metrics = evaluate(
        model = model,
        mode = args.mode,
        loader = loader,
        device = device,
        measure_timing = True,
    )
    metrics.update(eval_metrics)

    print(json.dumps(metrics, indent = 2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok = True)
        with open(args.out_json, "w", encoding = "utf-8") as f:
            json.dump(metrics, f, indent = 2)
        print(f"[OK] Wrote metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
