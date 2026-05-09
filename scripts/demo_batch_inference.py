#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rgbt_adaptive_fpn_lite_cal import CSRNetRGBT_AdaptiveFPNLiteCal


RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
T_MEAN = 0.485
T_STD = 0.229


def parse_args():
    ap = argparse.ArgumentParser(description="Run a lightweight demo inference grid for the final RGB-T model.")
    ap.add_argument("--data_root", default="data/RGBT-CC-CVPR2021")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/demo_results/demo_grid.png")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--img_h", type=int, default=768)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--tile_h", type=int, default=150)
    ap.add_argument("--tile_w", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return ap.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available.")
    return torch.device(name)


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    return {k.removeprefix("module."): v for k, v in state.items()}


def sample_ids(split_dir: Path, requested: List[str], num: int, seed: int) -> List[str]:
    if requested:
        return [str(x) for x in requested]
    ids = sorted(p.name.replace("_RGB.jpg", "").replace("_RGB.png", "") for p in split_dir.glob("*_RGB.*"))
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:num]


def load_gt_count(gt_path: Path) -> float:
    if not gt_path.exists():
        return float("nan")
    try:
        obj = json.loads(gt_path.read_text())
    except Exception:
        return float("nan")
    if isinstance(obj, dict):
        if "count" in obj:
            return float(obj["count"])
        pts = obj.get("points")
        if isinstance(pts, list):
            return float(len(pts))
    if isinstance(obj, list):
        return float(len(obj))
    return float("nan")


def rgb_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - RGB_MEAN) / RGB_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def thermal_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = ((arr - T_MEAN) / T_STD)[None, ...]
    return torch.from_numpy(arr).contiguous()


def display_image(path: Path, tile_h: int, tile_w: int, thermal: bool = False) -> Image.Image:
    mode = "L" if thermal else "RGB"
    return Image.open(path).convert(mode).convert("RGB").resize((tile_w, tile_h), Image.BILINEAR)


def density_heatmap(den: np.ndarray, tile_h: int, tile_w: int) -> Image.Image:
    den = np.asarray(den, dtype=np.float32)
    den = den - float(den.min())
    if float(den.max()) > 0:
        den = den / float(den.max())
    gray = (den * 255.0).clip(0, 255).astype(np.uint8)
    gray_img = Image.fromarray(gray, mode="L").resize((tile_w, tile_h), Image.BILINEAR)
    g = np.asarray(gray_img, dtype=np.uint8)
    heat = np.stack([g, np.maximum(g // 3, 40), 255 - g], axis=2)
    return Image.fromarray(heat, mode="RGB")


def font(size: int):
    for name in ("Arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


@torch.no_grad()
def main():
    args = parse_args()
    split_dir = Path(args.data_root) / args.split
    ids = sample_ids(split_dir, args.ids, args.num, args.seed)
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)
    print(f"[ids] {' '.join(ids)}", flush=True)

    model = CSRNetRGBT_AdaptiveFPNLiteCal(load_imagenet=False)
    missing, unexpected = model.load_state_dict(load_state(Path(args.checkpoint)), strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}", flush=True)
    model.to(device).eval()

    label_font = font(16)
    small_font = font(14)
    gap = 14
    header_h = 34
    row_h = args.tile_h + header_h
    cols = 3
    out_w = cols * args.tile_w + (cols - 1) * gap
    out_h = len(ids) * row_h + (len(ids) - 1) * gap
    canvas = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(canvas)

    summary = []
    for row, sid in enumerate(ids):
        rgb_path = split_dir / f"{sid}_RGB.jpg"
        t_path = split_dir / f"{sid}_T.jpg"
        gt_path = split_dir / f"{sid}_GT.json"
        if not rgb_path.exists() or not t_path.exists():
            print(f"[skip] missing files for {sid}", flush=True)
            continue

        x4 = torch.cat(
            [rgb_tensor(rgb_path, args.img_h, args.img_w), thermal_tensor(t_path, args.img_h, args.img_w)],
            dim=0,
        ).unsqueeze(0).to(device)
        den = model(x4)[0, 0].detach().float().cpu().numpy()
        pred = float(den.sum())
        gt = load_gt_count(gt_path)
        print(f"[result] {sid}: GT={gt:.0f} Pred={pred:.2f}", flush=True)
        summary.append({"id": sid, "gt_count": gt, "pred_count": pred})

        y = row * (row_h + gap)
        tiles: List[Tuple[str, Image.Image]] = [
            (f"{sid} RGB", display_image(rgb_path, args.tile_h, args.tile_w)),
            ("Thermal", display_image(t_path, args.tile_h, args.tile_w, thermal=True)),
            (f"Pred density  GT={gt:.0f}  Pred={pred:.1f}", density_heatmap(den, args.tile_h, args.tile_w)),
        ]
        for col, (title, img) in enumerate(tiles):
            x = col * (args.tile_w + gap)
            draw.text((x + 4, y + 8), title, fill=(20, 30, 45), font=label_font if col < 2 else small_font)
            canvas.paste(img, (x, y + header_h))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {out_path}", flush=True)
    print(f"[saved] {out_path.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
