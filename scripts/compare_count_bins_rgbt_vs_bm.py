#!/usr/bin/env python3
import argparse
import json
import os
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rgbt_cc import load_points
from scripts.eval_rgbt import build_model as build_student_model
from scripts.eval_rgbt import load_checkpoint as load_student_checkpoint


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    sid: str
    rgb_path: Path
    t_path: Path
    gt_path: Path
    gt_count: float


def parse_args():
    ap = argparse.ArgumentParser(description="Compare calibrated RGBT student vs BM/SOTA across count bins.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--student_mode", default="adaptive_fpn_lite_cal")
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--third_party_root", default=str(PROJECT_ROOT / "third_party" / "Broker-Modality-Crowd-Counting"))
    ap.add_argument("--bm_ckpt", required=True)
    ap.add_argument("--vgg19_local", required=True)
    ap.add_argument("--img_h", type=int, default=768)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--out_stride", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--per_bin", type=int, default=2)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def pick_existing(path_no_ext: Path, exts: Sequence[str]) -> Optional[Path]:
    for ext in exts:
        p = Path(str(path_no_ext) + ext)
        if p.exists():
            return p
    return None


def load_rgb_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).contiguous()


def load_t4_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = ((arr - 0.485) / 0.229)[None, ...]
    return torch.from_numpy(arr).contiguous()


def load_bm_rgb_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).contiguous()


def load_bm_t_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((w, h), Image.BILINEAR).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).contiguous()


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
            vgg_path = Path(vgg19_local).resolve()
            if not vgg_path.is_file():
                raise FileNotFoundError(f"Local VGG19 backbone not found: {vgg_path}")
            return torch_load_any(str(vgg_path))
        return orig(url, *args, **kwargs)

    model_zoo.load_url = patched


def select_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError("Unsupported checkpoint format for BM checkpoint")


def normalize_bm_state_dict_keys(state_dict):
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("feature."):
            key = "features." + key[len("feature."):]
        elif key.startswith("reg."):
            key = "reg_layer_0." + key[len("reg."):]
        remapped[key] = value
    return remapped


def build_bm_model(third_party_root: Path, ckpt_path: Path, vgg19_local: Path, device: torch.device):
    fine_tuning_root = third_party_root / "Fine-tuning"
    bm_file = fine_tuning_root / "models" / "bm.py"
    if not bm_file.is_file():
        raise FileNotFoundError(f"Official BM model file not found: {bm_file}")

    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    patch_vgg19_loader(str(vgg19_local))

    saved_model_modules = {
        k: v for k, v in list(sys.modules.items()) if k == "models" or k.startswith("models.")
    }
    for k in list(saved_model_modules.keys()):
        sys.modules.pop(k, None)
    if str(fine_tuning_root) in sys.path:
        sys.path.remove(str(fine_tuning_root))
    sys.path.insert(0, str(fine_tuning_root))

    try:
        BM = importlib.import_module("models.bm").BM  # type: ignore[attr-defined]
    finally:
        if str(fine_tuning_root) in sys.path:
            sys.path.remove(str(fine_tuning_root))
        for k in list(sys.modules.keys()):
            if k == "models" or k.startswith("models."):
                sys.modules.pop(k, None)
        sys.modules.update(saved_model_modules)

    model = BM().to(device)
    model.eval()

    raw = torch_load_any(str(ckpt_path))
    state_dict = normalize_bm_state_dict_keys(select_state_dict(raw))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("BM checkpoint did not load strictly.")
    return model


def build_samples(data_root: Path, split: str) -> List[Sample]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    rgb_names = sorted([p.name for p in split_dir.iterdir() if p.name.endswith("_RGB.jpg") or p.name.endswith("_RGB.png")])
    ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in rgb_names})
    out: List[Sample] = []
    for sid in ids:
        rgb_p = pick_existing(split_dir / f"{sid}_RGB", [".jpg", ".png"])
        t_p = pick_existing(split_dir / f"{sid}_T", [".jpg", ".png"])
        gt_p = pick_existing(split_dir / f"{sid}_GT", [".json", ".mat"])
        if rgb_p is None or t_p is None or gt_p is None:
            continue
        pts = load_points(str(gt_p))
        out.append(Sample(sid=sid, rgb_path=rgb_p, t_path=t_p, gt_path=gt_p, gt_count=float(len(pts))))
    if not out:
        raise RuntimeError(f"No valid samples found in {split_dir}")
    return out


def select_bins(samples: Sequence[Sample], per_bin: int) -> List[Tuple[str, List[Sample]]]:
    bins = [
        ("lt10", lambda c: c < 10),
        ("10to20", lambda c: 10 <= c < 20),
        ("20to50", lambda c: 20 <= c < 50),
        ("50to100", lambda c: 50 <= c < 100),
        ("gt100", lambda c: c >= 100),
    ]
    out = []
    for name, fn in bins:
        chosen = [s for s in samples if fn(s.gt_count)][:per_bin]
        if chosen:
            out.append((name, chosen))
    return out


@torch.no_grad()
def benchmark_student(model, x4: torch.Tensor, amp: bool, warmup: int, iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                _ = model(x4)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        times_ms = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        out = None
        for _ in range(iters):
            starter.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model(x4)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    return out, float(np.mean(times_ms)), float(torch.cuda.max_memory_allocated() / (1024 ** 2))


@torch.no_grad()
def benchmark_bm(model, rgb: torch.Tensor, t: torch.Tensor, amp: bool, warmup: int, iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                _ = model([rgb, t])
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        times_ms = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        out = None
        for _ in range(iters):
            starter.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model([rgb, t])
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    return out, float(np.mean(times_ms)), float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def density_to_heatmap(den: np.ndarray, out_hw: Tuple[int, int]) -> Image.Image:
    den = np.asarray(den, dtype=np.float32)
    den = den - den.min()
    if den.max() > 0:
        den = den / den.max()
    den = (255.0 * den).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(den, mode="L").resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    rgb = np.stack([arr, np.zeros_like(arr), 255 - arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def render_panel(sample: Sample, student_pred: float, bm_pred: float, student_den: np.ndarray, bm_den: np.ndarray, out_path: Path):
    rgb = Image.open(sample.rgb_path).convert("RGB").resize((320, 240), Image.BILINEAR)
    t = Image.open(sample.t_path).convert("L").convert("RGB").resize((320, 240), Image.BILINEAR)
    student_heat = density_to_heatmap(student_den, (240, 320))
    bm_heat = density_to_heatmap(bm_den, (240, 320))

    panel = Image.new("RGB", (1280, 320), color=(255, 255, 255))
    panel.paste(rgb, (0, 40))
    panel.paste(t, (320, 40))
    panel.paste(student_heat, (640, 40))
    panel.paste(bm_heat, (960, 40))

    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), f"{sample.sid}  GT={sample.gt_count:.0f}", fill=(0, 0, 0))
    draw.text((120, 285), "RGB", fill=(0, 0, 0))
    draw.text((430, 285), "Thermal", fill=(0, 0, 0))
    draw.text((655, 10), f"Student count={student_pred:.2f}", fill=(0, 0, 0))
    draw.text((975, 10), f"BM count={bm_pred:.2f}", fill=(0, 0, 0))
    draw.text((720, 285), "Student density", fill=(0, 0, 0))
    draw.text((1045, 285), "BM density", fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run on a GPU node.")

    device = torch.device("cuda")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    student = build_student_model(args.student_mode, load_imagenet=False).to(device)
    load_student_checkpoint(student, args.student_ckpt, device, strict=False)
    student.eval()

    bm = build_bm_model(
        Path(args.third_party_root).resolve(),
        Path(args.bm_ckpt).resolve(),
        Path(args.vgg19_local).resolve(),
        device,
    )

    samples = build_samples(Path(args.data_root).resolve(), args.split)
    chosen_bins = select_bins(samples, args.per_bin)

    results = {
        "student_mode": args.student_mode,
        "student_ckpt": str(Path(args.student_ckpt).resolve()),
        "bm_ckpt": str(Path(args.bm_ckpt).resolve()),
        "img_h": args.img_h,
        "img_w": args.img_w,
        "amp": bool(args.amp),
        "warmup": args.warmup,
        "iters": args.iters,
        "bins": [],
    }

    for bin_name, items in chosen_bins:
        bin_rows = []
        for sample in items:
            x_rgb = load_rgb_tensor(sample.rgb_path, args.img_h, args.img_w)
            x_t1 = load_t4_tensor(sample.t_path, args.img_h, args.img_w)
            x4 = torch.cat([x_rgb, x_t1], dim=0).unsqueeze(0).to(device, non_blocking=True)

            bm_rgb = load_bm_rgb_tensor(sample.rgb_path, args.img_h, args.img_w).unsqueeze(0).to(device, non_blocking=True)
            bm_t = load_bm_t_tensor(sample.t_path, args.img_h, args.img_w).unsqueeze(0).to(device, non_blocking=True)

            student_out, student_ms, student_mem = benchmark_student(student, x4, args.amp, args.warmup, args.iters)
            bm_out, bm_ms, bm_mem = benchmark_bm(bm, bm_rgb, bm_t, args.amp, args.warmup, args.iters)

            student_den = student_out.detach().float().cpu().numpy()[0, 0]
            bm_den = bm_out.detach().float().cpu().numpy()[0, 0]
            student_cnt = float(student_den.sum())
            bm_cnt = float(bm_den.sum())

            panel_path = out_dir / "panels" / bin_name / f"{sample.sid}.png"
            render_panel(sample, student_cnt, bm_cnt, student_den, bm_den, panel_path)

            row = {
                "id": sample.sid,
                "gt_count": sample.gt_count,
                "student_count": student_cnt,
                "bm_count": bm_cnt,
                "student_latency_ms": student_ms,
                "bm_latency_ms": bm_ms,
                "student_fps": float(1000.0 / student_ms),
                "bm_fps": float(1000.0 / bm_ms),
                "student_peak_mem_mb": student_mem,
                "bm_peak_mem_mb": bm_mem,
                "panel_path": str(panel_path),
            }
            bin_rows.append(row)

        results["bins"].append(
            {
                "name": bin_name,
                "num_images": len(bin_rows),
                "student_avg_latency_ms": float(np.mean([r["student_latency_ms"] for r in bin_rows])),
                "bm_avg_latency_ms": float(np.mean([r["bm_latency_ms"] for r in bin_rows])),
                "student_avg_fps": float(np.mean([r["student_fps"] for r in bin_rows])),
                "bm_avg_fps": float(np.mean([r["bm_fps"] for r in bin_rows])),
                "rows": bin_rows,
            }
        )

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[json] wrote {json_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
