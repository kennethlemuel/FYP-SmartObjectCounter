#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inference-only benchmark for Broker Modality BM")
    p.add_argument("--repo_root", default=None)
    p.add_argument("--third_party_root", default=None)
    p.add_argument("--ckpt", default="")
    p.add_argument("--vgg19_local", default="")
    p.add_argument("--h", type=int, default=768)
    p.add_argument("--w", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--rgb", default="")
    p.add_argument("--t", default="")
    p.add_argument("--allow_nonstrict", action="store_true")
    p.add_argument("--json_out", default="")
    return p.parse_args()


def infer_repo_root(args):
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parents[1]


def infer_third_party_root(repo_root: Path, args):
    if args.third_party_root:
        return Path(args.third_party_root).resolve()
    return repo_root / "third_party" / "Broker-Modality-Crowd-Counting"


def torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def patch_vgg19_loader(vgg19_local: str):
    import torch.utils.model_zoo as model_zoo

    orig = model_zoo.load_url

    def patched(url, *args, **kwargs):
        if url.endswith("vgg19-dcbb9e9d.pth"):
            if vgg19_local:
                vgg_path = Path(vgg19_local).resolve()
                if not vgg_path.is_file():
                    raise FileNotFoundError(f"Local VGG19 backbone not found: {vgg_path}")
                print(f"[init] using local VGG19 backbone: {vgg_path}")
                return torch_load_any(str(vgg_path))
            print("[init] no local VGG19 backbone provided; returning empty state_dict")
            return {}
        return orig(url, *args, **kwargs)

    model_zoo.load_url = patched


def select_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], key
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, "raw_state_dict"
    raise RuntimeError(
        "Unsupported checkpoint format. Expected raw state_dict or dict containing "
        "model_state_dict/state_dict/model/net."
    )


def strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_image_as_tensor(path: str, h: int, w: int):
    img = Image.open(path).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def build_inputs(args, device):
    if args.rgb or args.t:
        if not (args.rgb and args.t):
            raise RuntimeError("Provide both --rgb and --t together.")
        rgb = load_image_as_tensor(args.rgb, args.h, args.w).unsqueeze(0)
        t = load_image_as_tensor(args.t, args.h, args.w).unsqueeze(0)
        x_rgb = rgb.repeat(args.batch_size, 1, 1, 1).to(device, non_blocking=True)
        x_t = t.repeat(args.batch_size, 1, 1, 1).to(device, non_blocking=True)
        return x_rgb, x_t, "real_pair"
    x_rgb = torch.randn(args.batch_size, 3, args.h, args.w, device=device)
    x_t = torch.randn(args.batch_size, 3, args.h, args.w, device=device)
    return x_rgb, x_t, "synthetic"


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on a GPU node.")
    if args.h % 8 != 0 or args.w % 8 != 0:
        raise RuntimeError(f"Input size must be divisible by 8, got {args.h}x{args.w}.")

    repo_root = infer_repo_root(args)
    third_party_root = infer_third_party_root(repo_root, args)
    fine_tuning_root = third_party_root / "Fine-tuning"
    bm_file = fine_tuning_root / "models" / "bm.py"
    if not bm_file.is_file():
        raise FileNotFoundError(f"Official BM model file not found: {bm_file}")

    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    patch_vgg19_loader(args.vgg19_local)

    sys.path.insert(0, str(fine_tuning_root))
    from models.bm import BM

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = BM().to(device)
    model.eval()

    ckpt_info = {
        "used": False,
        "path": "",
        "source_key": "",
        "strict": True,
        "missing": [],
        "unexpected": [],
    }

    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        raw = torch_load_any(str(ckpt_path))
        state_dict, source_key = select_state_dict(raw)
        state_dict = strip_module_prefix(state_dict)
        try:
            incompatible = model.load_state_dict(state_dict, strict=True)
            strict_used = True
        except RuntimeError as e:
            print(f"[ckpt] strict load failed: {e}", file=sys.stderr)
            if not args.allow_nonstrict:
                raise
            incompatible = model.load_state_dict(state_dict, strict=False)
            strict_used = False
        ckpt_info["used"] = True
        ckpt_info["path"] = str(ckpt_path)
        ckpt_info["source_key"] = source_key
        ckpt_info["strict"] = strict_used
        ckpt_info["missing"] = list(incompatible.missing_keys)
        ckpt_info["unexpected"] = list(incompatible.unexpected_keys)
        print(
            f"[ckpt] loaded {ckpt_path} source_key={source_key} "
            f"strict={strict_used} missing={len(ckpt_info['missing'])} "
            f"unexpected={len(ckpt_info['unexpected'])}"
        )
    else:
        print("[ckpt] none provided")

    params_total, params_trainable = count_parameters(model)
    x_rgb, x_t, input_mode = build_inputs(args, device)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            y = model([x_rgb, x_t])
    torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                _ = model([x_rgb, x_t])
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        times_ms = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(args.iters):
            starter.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                _ = model([x_rgb, x_t])
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))

    summary = {
        "model": "BM",
        "repo_root": str(repo_root),
        "third_party_root": str(third_party_root),
        "input_mode": input_mode,
        "batch_size": args.batch_size,
        "img_h": args.h,
        "img_w": args.w,
        "precision": "amp_fp16" if args.amp else "fp32",
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_latency_ms": float(np.mean(times_ms)),
        "std_latency_ms": float(np.std(times_ms)),
        "throughput_img_s": float(args.batch_size * 1000.0 / np.mean(times_ms)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "output_shape": list(y.shape),
        "output_sum": float(y.sum().item()),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": torch.__version__,
        "ckpt": ckpt_info,
    }

    print("")
    print("=== BM Inference Benchmark Summary ===")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k2, v2 in value.items():
                print(f"  {k2}: {v2}")
        else:
            print(f"{key}: {value}")

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[json] wrote {json_path}")


if __name__ == "__main__":
    main()
