import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import scipy.io as sio
except Exception:
    sio = None


# -----------------------------
# Determinism helpers
# -----------------------------
def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker seeding. Use this with:
      DataLoader(..., worker_init_fn = seed_worker, generator = g)

    So random ops inside workers (if any) are deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# Point / density utilities (CSRNet-style)
# -----------------------------
def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy = False)


def _safe_imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _safe_imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    # If 3-channel thermal accidentally stored, convert to gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def _normalize_uint_like_to_01(x: np.ndarray) -> np.ndarray:
    """
    Normalize an integer-like image to [0, 1] robustly.
    Handles 8-bit, 16-bit, or arbitrary ranges.
    """
    x = x.astype(np.float32, copy = False)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min + 1e-6:
        return np.zeros_like(x, dtype = np.float32)
    return (x - x_min) / (x_max - x_min)


def _normalize_imagenet(chw: torch.Tensor) -> torch.Tensor:
    """
    ImageNet normalization for 3-channel inputs. Assumes input in [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype = chw.dtype, device = chw.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype = chw.dtype, device = chw.device)[:, None, None]
    return (chw - mean) / std


def _normalize_4ch(chw: torch.Tensor) -> torch.Tensor:
    """
    Normalize 4-channel early-fusion input: RGB (ImageNet) + Thermal (0.5/0.5 default).
    Assumes input in [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5], dtype = chw.dtype, device = chw.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225, 0.5], dtype = chw.dtype, device = chw.device)[:, None, None]
    return (chw - mean) / std


def _mat_to_points(mat_obj: Dict) -> np.ndarray:
    """
    Try multiple common keys to extract Nx2 points from a .mat.
    """
    # Common keys seen across crowd datasets
    candidate_keys = [
        "annPoints",
        "image_info",
        "gt",
        "points",
        "p",
        "loc",
        "location",
    ]

    for k in candidate_keys:
        if k not in mat_obj:
            continue

        v = mat_obj[k]

        # ShanghaiTech-style image_info
        if k == "image_info":
            try:
                v = v[0, 0][0, 0][0]
            except Exception:
                pass

        pts = np.array(v)

        # Some mats store as 2xN
        if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T

        if pts.ndim == 2 and pts.shape[1] == 2:
            return _as_float32(pts)

    raise KeyError(f"Could not find points in .mat keys: {list(mat_obj.keys())}")


def load_points(gt_path: str) -> np.ndarray:
    """
    Load Nx2 points from:
      - .mat (common crowd dataset formats)
      - .json (expects list of [x,y] or dicts with x/y)
    """
    ext = os.path.splitext(gt_path)[1].lower()

    if ext == ".json":
        with open(gt_path, "r") as f:
            data = json.load(f)
        pts: List[List[float]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and ("x" in item) and ("y" in item):
                    pts.append([float(item["x"]), float(item["y"])])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pts.append([float(item[0]), float(item[1])])
        return np.asarray(pts, dtype = np.float32)

    if ext == ".mat":
        if sio is None:
            raise ImportError("scipy is required to load .mat ground-truth files.")
        mat = sio.loadmat(gt_path)
        return _mat_to_points(mat)

    raise ValueError(f"Unsupported GT extension: {ext} ({gt_path})")


def density_from_points(
    pts_xy: np.ndarray,
    out_h: int,
    out_w: int,
    sigma: float = 4.0,
) -> np.ndarray:
    """
    CSRNet-style: place Gaussian at each head point on an output grid.

    Important property for meaningful training loss:
      sum(density) ~= number of points  (after normalization below).

    We normalize the density map to exactly preserve count (unless empty).
    """
    density = np.zeros((out_h, out_w), dtype = np.float32)
    if pts_xy is None or len(pts_xy) == 0:
        return density

    pts = np.asarray(pts_xy, dtype = np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts_xy must be Nx2, got shape {pts.shape}")

    # Precompute gaussian kernel size (odd)
    ksize = int(6.0 * sigma + 1.0)
    if ksize % 2 == 0:
        ksize += 1
    radius = ksize // 2

    # Build a gaussian kernel once
    ax = np.arange(-radius, radius + 1, dtype = np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma)).astype(np.float32)

    # Add kernels
    for (x, y) in pts:
        x_i = int(round(x))
        y_i = int(round(y))
        if x_i < 0 or x_i >= out_w or y_i < 0 or y_i >= out_h:
            continue

        x1 = max(0, x_i - radius)
        y1 = max(0, y_i - radius)
        x2 = min(out_w, x_i + radius + 1)
        y2 = min(out_h, y_i + radius + 1)

        kx1 = x1 - (x_i - radius)
        ky1 = y1 - (y_i - radius)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)

        density[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    # Normalize to preserve count
    s = float(density.sum())
    if s > 1e-6:
        density *= (len(pts) / s)

    return density


# -----------------------------
# Dataset configs
# -----------------------------
@dataclass
class RGBTCCConfig:
    root_dir: str
    split: str
    img_size: Tuple[int, int] = (224, 224)  # (H, W)
    out_stride: int = 8
    gt_preference: Tuple[str, ...] = (".mat", ".json")  # try in this order


def _list_sorted_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            files.append(os.path.join(folder, f))
    files.sort()
    return files


def _find_gt_path(gt_dir: str, base_name: str, exts: Tuple[str, ...]) -> str:
    for ext in exts:
        cand = os.path.join(gt_dir, base_name + ext)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No GT found for {base_name} in {gt_dir} with exts {exts}")


def _resize_and_scale_points(
    img: np.ndarray,
    pts: np.ndarray,
    new_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image to new_hw = (H, W) and scale points accordingly.
    Points expected in pixel coordinates (x,y) with origin at top-left.
    """
    h0, w0 = img.shape[:2]
    new_h, new_w = new_hw

    img_rs = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR)

    if pts is None or len(pts) == 0:
        return img_rs, np.zeros((0, 2), dtype = np.float32)

    pts = np.asarray(pts, dtype = np.float32)
    sx = float(new_w) / float(w0)
    sy = float(new_h) / float(h0)
    pts_rs = pts.copy()
    pts_rs[:, 0] *= sx
    pts_rs[:, 1] *= sy
    return img_rs, pts_rs


def _to_chw_tensor_rgb(img_rgb_uint: np.ndarray) -> torch.Tensor:
    """
    img_rgb_uint: HxWx3 uint-like or float, converted to float [0,1] then CHW tensor.
    """
    if img_rgb_uint.dtype != np.float32:
        img = img_rgb_uint.astype(np.float32) / 255.0
    else:
        img = img_rgb_uint
        if img.max() > 1.5:
            img = img / 255.0

    chw = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return chw


def _to_chw_tensor_gray01(img_gray: np.ndarray) -> torch.Tensor:
    """
    img_gray: HxW uint/float. Normalize to [0,1], output 1xHxW.
    """
    if img_gray.dtype == np.uint8:
        x = img_gray.astype(np.float32) / 255.0
    elif img_gray.dtype in (np.uint16, np.int16, np.int32, np.uint32, np.int64, np.uint64):
        x = _normalize_uint_like_to_01(img_gray)
    else:
        x = img_gray.astype(np.float32, copy = False)
        if x.max() > 1.5:
            # assume 0..255-ish float
            x = x / 255.0
        else:
            # already 0..1-ish
            x = np.clip(x, 0.0, 1.0)

    x = torch.from_numpy(x)[None, ...].contiguous()
    return x


# -----------------------------
# Dataset variants used by train_rgbt_patched.py
# -----------------------------
class RGBTCC_PairedDatasetPoints(Dataset):
    """
    Returns (rgb_3ch, t_3ch, pts_xy, meta_dict)

    - rgb_3ch and t_3ch are normalized tensors
    - pts_xy are points in resized image coordinates (x,y)
    - No random augmentation here. Pairwise random aug should be done in train wrapper.
    """

    def __init__(self, cfg: RGBTCCConfig):
        self.cfg = cfg
        self.rgb_dir = os.path.join(cfg.root_dir, cfg.split, "RGB")
        self.t_dir = os.path.join(cfg.root_dir, cfg.split, "T")
        self.gt_dir = os.path.join(cfg.root_dir, cfg.split, "GT")

        self.rgb_paths = _list_sorted_images(self.rgb_dir)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        base = os.path.splitext(os.path.basename(rgb_path))[0]

        # Thermal path tries same base name
        t_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            cand = os.path.join(self.t_dir, base + ext)
            if os.path.isfile(cand):
                t_path = cand
                break
        if t_path is None:
            raise FileNotFoundError(f"No thermal image found for base={base} in {self.t_dir}")

        gt_path = _find_gt_path(self.gt_dir, base, self.cfg.gt_preference)

        rgb = _safe_imread_rgb(rgb_path)
        t = _safe_imread_gray(t_path)
        pts = load_points(gt_path)

        # Resize + scale points consistently (paired)
        rgb_rs, pts_rs = _resize_and_scale_points(rgb, pts, self.cfg.img_size)

        # Thermal resize (no points scaling needed, already done)
        t_rs = cv2.resize(t, (self.cfg.img_size[1], self.cfg.img_size[0]), interpolation = cv2.INTER_LINEAR)

        # Convert to tensors
        rgb_t = _normalize_imagenet(_to_chw_tensor_rgb(rgb_rs))
        t_1 = _to_chw_tensor_gray01(t_rs)
        t_3 = t_1.repeat(3, 1, 1)  # to 3ch for late/adaptive models
        t_t = _normalize_imagenet(t_3)

        pts_t = torch.from_numpy(pts_rs.astype(np.float32))

        meta = {
            "rgb_path": rgb_path,
            "t_path": t_path,
            "gt_path": gt_path,
            "base": base,
        }

        return rgb_t, t_t, pts_t, meta


class RGBTCC_ThermalDatasetPoints(Dataset):
    """
    Returns (t_3ch, pts_xy, meta_dict)
    Useful for thermal-only baseline (keeps same point protocol).
    """

    def __init__(self, cfg: RGBTCCConfig):
        self.cfg = cfg
        self.t_dir = os.path.join(cfg.root_dir, cfg.split, "T")
        self.rgb_dir = os.path.join(cfg.root_dir, cfg.split, "RGB")
        self.gt_dir = os.path.join(cfg.root_dir, cfg.split, "GT")

        # Use RGB list as anchor for matching GT names
        self.rgb_paths = _list_sorted_images(self.rgb_dir)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        base = os.path.splitext(os.path.basename(rgb_path))[0]

        t_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            cand = os.path.join(self.t_dir, base + ext)
            if os.path.isfile(cand):
                t_path = cand
                break
        if t_path is None:
            raise FileNotFoundError(f"No thermal image found for base={base} in {self.t_dir}")

        gt_path = _find_gt_path(self.gt_dir, base, self.cfg.gt_preference)

        t = _safe_imread_gray(t_path)
        pts = load_points(gt_path)

        # Need resize reference for point scaling, use RGB size
        rgb = _safe_imread_rgb(rgb_path)
        rgb_rs, pts_rs = _resize_and_scale_points(rgb, pts, self.cfg.img_size)

        t_rs = cv2.resize(t, (self.cfg.img_size[1], self.cfg.img_size[0]), interpolation = cv2.INTER_LINEAR)

        t_1 = _to_chw_tensor_gray01(t_rs)
        t_3 = t_1.repeat(3, 1, 1)
        t_t = _normalize_imagenet(t_3)

        pts_t = torch.from_numpy(pts_rs.astype(np.float32))

        meta = {
            "t_path": t_path,
            "gt_path": gt_path,
            "base": base,
        }

        return t_t, pts_t, meta


class RGBTCC_RGBDatasetPoints(Dataset):
    """
    Returns (rgb_3ch, pts_xy, meta_dict)
    Useful for RGB-only baseline.
    """

    def __init__(self, cfg: RGBTCCConfig):
        self.cfg = cfg
        self.rgb_dir = os.path.join(cfg.root_dir, cfg.split, "RGB")
        self.gt_dir = os.path.join(cfg.root_dir, cfg.split, "GT")
        self.rgb_paths = _list_sorted_images(self.rgb_dir)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        gt_path = _find_gt_path(self.gt_dir, base, self.cfg.gt_preference)

        rgb = _safe_imread_rgb(rgb_path)
        pts = load_points(gt_path)

        rgb_rs, pts_rs = _resize_and_scale_points(rgb, pts, self.cfg.img_size)

        rgb_t = _normalize_imagenet(_to_chw_tensor_rgb(rgb_rs))
        pts_t = torch.from_numpy(pts_rs.astype(np.float32))

        meta = {"rgb_path": rgb_path, "gt_path": gt_path, "base": base}
        return rgb_t, pts_t, meta


class RGBTCC_EarlyFusionDatasetPoints(Dataset):
    """
    Returns (rgbt_4ch, pts_xy, meta_dict)
    Early fusion expects a 4-channel tensor: [R,G,B,T].
    """

    def __init__(self, cfg: RGBTCCConfig):
        self.cfg = cfg
        self.rgb_dir = os.path.join(cfg.root_dir, cfg.split, "RGB")
        self.t_dir = os.path.join(cfg.root_dir, cfg.split, "T")
        self.gt_dir = os.path.join(cfg.root_dir, cfg.split, "GT")

        self.rgb_paths = _list_sorted_images(self.rgb_dir)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        base = os.path.splitext(os.path.basename(rgb_path))[0]

        t_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            cand = os.path.join(self.t_dir, base + ext)
            if os.path.isfile(cand):
                t_path = cand
                break
        if t_path is None:
            raise FileNotFoundError(f"No thermal image found for base={base} in {self.t_dir}")

        gt_path = _find_gt_path(self.gt_dir, base, self.cfg.gt_preference)

        rgb = _safe_imread_rgb(rgb_path)
        t = _safe_imread_gray(t_path)
        pts = load_points(gt_path)

        rgb_rs, pts_rs = _resize_and_scale_points(rgb, pts, self.cfg.img_size)
        t_rs = cv2.resize(t, (self.cfg.img_size[1], self.cfg.img_size[0]), interpolation = cv2.INTER_LINEAR)

        rgb_chw = _to_chw_tensor_rgb(rgb_rs)
        t_1 = _to_chw_tensor_gray01(t_rs)  # 1xHxW

        # Concatenate 4 channels and normalize
        rgbt_4 = torch.cat([rgb_chw, t_1], dim = 0)
        rgbt_4 = _normalize_4ch(rgbt_4)

        pts_t = torch.from_numpy(pts_rs.astype(np.float32))

        meta = {"rgb_path": rgb_path, "t_path": t_path, "gt_path": gt_path, "base": base}
        return rgbt_4, pts_t, meta
