#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rgbt_cc import density_from_points, load_points
from scripts.eval_rgbt import build_model, load_checkpoint


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    sid: str
    rgb_path: Path
    t_path: Path
    gt_path: Path


@dataclass
class ModelSpec:
    label: str
    mode: str
    ckpt: str


def parse_args():
    ap = argparse.ArgumentParser(description="Export report-ready qualitative panels from in-house checkpoints.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--img_h", type=int, default=768)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--out_stride", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--tile_w", type=int, default=220)
    ap.add_argument("--tile_h", type=int, default=160)
    ap.add_argument("--col_gap", type=int, default=16)
    ap.add_argument("--row_gap", type=int, default=18)
    ap.add_argument("--figure_name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_ids", nargs="+", required=True)
    ap.add_argument(
        "--model_spec",
        action="append",
        default=[],
        help="Repeated. Format: LABEL|MODE|CKPT_PATH",
    )
    ap.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional repeated row note. Format: IMAGE_ID|free text",
    )
    ap.add_argument("--include_gt", action="store_true")
    return ap.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def display_label(label: str) -> str:
    mapping = {
        "GT density": "Ground Truth Density",
        "rgbt_base": "RGBT Base",
        "rgbt_early": "RGBT Early",
        "adaptive_fpn_lite": "Adaptive FPN (Lite)",
        "Adaptive FPN (calibrated)": "Adaptive FPN (Lite + Cal)",
    }
    return mapping.get(label, label)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:
        return draw.textsize(text, font=font)


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    shadow: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = xy
    for dx, dy in ((1, 1),):
        draw.text((x + dx, y + dy), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def pick_existing(path_no_ext: Path, exts: Sequence[str]) -> Path:
    for ext in exts:
        p = Path(str(path_no_ext) + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing file for stem {path_no_ext}")


def parse_model_specs(items: Sequence[str]) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for item in items:
        parts = item.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid --model_spec: {item}")
        out.append(ModelSpec(label=parts[0], mode=parts[1], ckpt=parts[2]))
    if not out:
        raise ValueError("At least one --model_spec is required.")
    return out


def parse_notes(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        parts = item.split("|", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid --note: {item}")
        out[parts[0]] = parts[1]
    return out


def build_sample_map(data_root: Path, split: str) -> Dict[str, Sample]:
    split_dir = data_root / split
    out: Dict[str, Sample] = {}
    for p in split_dir.iterdir():
        if p.name.endswith("_RGB.jpg") or p.name.endswith("_RGB.png"):
            sid = p.name.replace("_RGB.jpg", "").replace("_RGB.png", "")
            try:
                rgb_p = pick_existing(split_dir / f"{sid}_RGB", [".jpg", ".png"])
                t_p = pick_existing(split_dir / f"{sid}_T", [".jpg", ".png"])
                gt_p = pick_existing(split_dir / f"{sid}_GT", [".json", ".mat"])
            except FileNotFoundError:
                continue
            out[sid] = Sample(sid=sid, rgb_path=rgb_p, t_path=t_p, gt_path=gt_p)
    if not out:
        raise RuntimeError(f"No valid samples found in {split_dir}")
    return out


def load_rgb_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).contiguous()


def load_t1_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = ((arr - 0.485) / 0.229)[None, ...]
    return torch.from_numpy(arr).contiguous()


def load_t3_tensor(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("L").convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).contiguous()


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


def load_display_rgb(path: Path, out_hw: Tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize((out_hw[1], out_hw[0]), Image.BILINEAR)


def load_display_t(path: Path, out_hw: Tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("L").convert("RGB").resize((out_hw[1], out_hw[0]), Image.BILINEAR)


def gt_density_heatmap(sample: Sample, img_h: int, img_w: int, out_stride: int, sigma: float, tile_hw: Tuple[int, int]) -> Tuple[Image.Image, float]:
    pts = load_points(str(sample.gt_path))
    gt_count = float(len(pts))
    if pts.size > 0:
        rgb = Image.open(sample.rgb_path).convert("RGB")
        w0, h0 = rgb.size
        pts = pts.copy()
        pts[:, 0] *= (img_w / float(w0))
        pts[:, 1] *= (img_h / float(h0))
        pts[:, 0] /= out_stride
        pts[:, 1] /= out_stride
    den = density_from_points(
        pts,
        img_h // out_stride,
        img_w // out_stride,
        sigma=max(1.0, sigma / out_stride),
    )
    return density_to_heatmap(den, tile_hw), gt_count


@torch.no_grad()
def infer_density(model, mode: str, rgb_t: torch.Tensor, t1_t: torch.Tensor, t3_t: torch.Tensor, device: torch.device) -> np.ndarray:
    mode = mode.lower()
    model.eval()
    if mode == "rgb":
        out = model(rgb_t.unsqueeze(0).to(device, non_blocking=True))
    elif mode == "t":
        out = model(t3_t.unsqueeze(0).to(device, non_blocking=True))
    elif mode in ["base", "adaptive_fpn_lite", "adaptive_fpn_lite_cal"]:
        x4 = torch.cat([rgb_t, t1_t], dim=0).unsqueeze(0).to(device, non_blocking=True)
        out = model(x4)
    elif mode in ["early", "late", "adaptive_late"]:
        out = model(
            rgb_t.unsqueeze(0).to(device, non_blocking=True),
            t3_t.unsqueeze(0).to(device, non_blocking=True),
        )
    else:
        raise ValueError(f"Unsupported mode for qualitative export: {mode}")

    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().cpu().numpy()[0, 0]


def render_figure(
    samples: Sequence[Sample],
    notes: Dict[str, str],
    display_tiles: Dict[str, Tuple[Image.Image, Image.Image, Image.Image, float]],
    predictions: Dict[str, Dict[str, Tuple[np.ndarray, float]]],
    model_specs: Sequence[ModelSpec],
    tile_w: int,
    tile_h: int,
    col_gap: int,
    row_gap: int,
    include_gt: bool,
    out_path: Path,
):
    title_font = load_font(16)
    body_font = load_font(18)
    note_font = load_font(18)

    header_h = 42
    note_h = 18
    row_extra = note_h if notes else 0
    cols = 2 + len(model_specs) + (1 if include_gt else 0)
    row_h = header_h + tile_h + row_extra
    total_h = row_h * len(samples) + row_gap * max(0, len(samples) - 1)
    total_w = cols * tile_w + col_gap * max(0, cols - 1)

    panel = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    for ridx, sample in enumerate(samples):
        y0 = ridx * (row_h + row_gap)
        rgb_img, t_img, gt_heat, gt_count = display_tiles[sample.sid]

        tiles: List[Tuple[str, Image.Image, str]] = [
            ("RGB", rgb_img, ""),
            ("Thermal", t_img, ""),
        ]
        if include_gt:
            tiles.append(("GT density", gt_heat, f"GT count={gt_count:.0f}"))

        for spec in model_specs:
            den, cnt = predictions[sample.sid][spec.label]
            tiles.append((display_label(spec.label), density_to_heatmap(den, (tile_h, tile_w)), f"Pred={cnt:.2f}"))

        for cidx, (label, tile_img, sublabel) in enumerate(tiles):
            x0 = cidx * (tile_w + col_gap)
            draw.text((x0 + 8, y0 + 8), display_label(label), fill=(0, 0, 0), font=title_font)
            panel.paste(tile_img, (x0, y0 + header_h))
            if sublabel:
                draw_text_with_shadow(
                    draw,
                    (x0 + 8, y0 + header_h + tile_h - 28),
                    sublabel,
                    body_font,
                    fill=(255, 255, 255),
                )

        row_text = f"Image {sample.sid}  GT={gt_count:.0f}"
        draw_text_with_shadow(
            draw,
            (8, y0 + header_h + tile_h - 28),
            row_text,
            body_font,
            fill=(255, 255, 255),
        )
        note = notes.get(sample.sid)
        if note:
            draw.text((8, y0 + header_h + tile_h + 2), note, fill=(60, 60, 60), font=note_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for qualitative export. Run on a GPU node.")

    model_specs = parse_model_specs(args.model_spec)
    notes = parse_notes(args.note)
    sample_map = build_sample_map(Path(args.data_root).resolve(), args.split)
    samples = [sample_map[sid] for sid in args.image_ids]
    device = torch.device("cuda")

    models = []
    for spec in model_specs:
        model = build_model(spec.mode, load_imagenet=False).to(device)
        load_checkpoint(model, spec.ckpt, device, strict=False)
        model.eval()
        models.append((spec, model))

    display_tiles: Dict[str, Tuple[Image.Image, Image.Image, Image.Image, float]] = {}
    predictions: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = {}

    for sample in samples:
        rgb_disp = load_display_rgb(sample.rgb_path, (args.tile_h, args.tile_w))
        t_disp = load_display_t(sample.t_path, (args.tile_h, args.tile_w))
        gt_heat, gt_count = gt_density_heatmap(
            sample,
            args.img_h,
            args.img_w,
            args.out_stride,
            args.sigma,
            (args.tile_h, args.tile_w),
        )
        display_tiles[sample.sid] = (rgb_disp, t_disp, gt_heat, gt_count)

        rgb_t = load_rgb_tensor(sample.rgb_path, args.img_h, args.img_w)
        t1_t = load_t1_tensor(sample.t_path, args.img_h, args.img_w)
        t3_t = load_t3_tensor(sample.t_path, args.img_h, args.img_w)

        predictions[sample.sid] = {}
        for spec, model in models:
            den = infer_density(model, spec.mode, rgb_t, t1_t, t3_t, device)
            predictions[sample.sid][spec.label] = (den, float(den.sum()))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / f"{args.figure_name}.png"
    summary_path = out_dir / f"{args.figure_name}.json"

    render_figure(
        samples,
        notes,
        display_tiles,
        predictions,
        model_specs,
        args.tile_w,
        args.tile_h,
        args.col_gap,
        args.row_gap,
        bool(args.include_gt),
        figure_path,
    )

    summary = {
        "figure_name": args.figure_name,
        "image_ids": args.image_ids,
        "model_specs": [spec.__dict__ for spec in model_specs],
        "notes": notes,
        "rows": [],
    }
    for sample in samples:
        row = {"id": sample.sid, "gt_count": display_tiles[sample.sid][3], "predictions": {}}
        for spec in model_specs:
            _, cnt = predictions[sample.sid][spec.label]
            row["predictions"][spec.label] = {"mode": spec.mode, "count": cnt}
        summary["rows"].append(row)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[figure] {figure_path}")
    print(f"[json] {summary_path}")


if __name__ == "__main__":
    main()
