import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_tf_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = _IMAGENET_MEAN, std = _IMAGENET_STD),
])

_tf_t3 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = _IMAGENET_MEAN, std = _IMAGENET_STD),
])

_tf_t1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485], std = [0.229]),
])


def _pick_existing(path_no_ext, exts):
    for e in exts:
        p = path_no_ext + e
        if os.path.exists(p):
            return p
    return None


def _den_to_tensor(den_hw):
    den_hw = np.ascontiguousarray(den_hw.astype(np.float32, copy = False))
    return torch.tensor(den_hw, dtype = torch.float32).unsqueeze(0).contiguous()


def density_from_points(points_xy, h, w, sigma = 15.0):
    """
    points_xy: (N,2) in (x,y) coordinates in the SAME space as (h,w).
    Returns a density map normalized so sum == N (when N > 0).
    """
    dm = np.zeros((h, w), dtype = np.float32)
    if points_xy.size == 0:
        return dm

    xs = np.clip(points_xy[:, 0].astype(int), 0, w - 1)
    ys = np.clip(points_xy[:, 1].astype(int), 0, h - 1)

    dm[ys, xs] = 1.0
    dm = gaussian_filter(dm, sigma = sigma, mode = "constant")

    s = float(dm.sum())
    if s > 0:
        dm *= (len(xs) / s)
    return dm


def _load_points_json(p):
    with open(p, "r") as f:
        data = json.load(f)

    for k in ["points", "keypoints", "annotations", "labels"]:
        if k in data and isinstance(data[k], list):
            pts = np.array(data[k], dtype = np.float32).reshape(-1, 2)
            return pts

    return np.zeros((0, 2), dtype = np.float32)


def _load_points_mat(p):
    m = loadmat(p)

    if "point" in m:
        return np.array(m["point"], dtype = np.float32).reshape(-1, 2)

    if "image_info" in m:
        pts = m["image_info"][0, 0][0, 0][0]
        return np.array(pts, dtype = np.float32).reshape(-1, 2)

    return np.zeros((0, 2), dtype = np.float32)


def load_points(label_path_no_ext):
    json_p = label_path_no_ext + ".json"
    mat_p = label_path_no_ext + ".mat"

    if os.path.exists(json_p):
        return _load_points_json(json_p)
    if os.path.exists(mat_p):
        return _load_points_mat(mat_p)

    return np.zeros((0, 2), dtype = np.float32)


def seed_worker(worker_id: int) -> None:
    """Deterministic DataLoader worker seeding."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _to_chw_tensor_rgb(bgr_u8: np.ndarray) -> torch.Tensor:
    """OpenCV BGR uint8 -> CHW float32 RGB tensor in [0, 1]."""
    rgb = bgr_u8[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw)


def _to_chw_tensor_gray01(gray_u8: np.ndarray) -> torch.Tensor:
    """OpenCV gray uint8 -> 1xHxW float32 tensor in [0, 1]."""
    g = gray_u8.astype(np.float32) / 255.0
    return torch.from_numpy(g[None, ...])


def _normalize_imagenet(rgb_chw_01: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization for 3-channel CHW tensors in [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb_chw_01.dtype, device=rgb_chw_01.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=rgb_chw_01.dtype, device=rgb_chw_01.device)[:, None, None]
    return (rgb_chw_01 - mean) / std


def _to_t3(img_any):
    if img_any.ndim == 2:
        g = img_any
    else:
        g = cv2.cvtColor(img_any, cv2.COLOR_BGR2GRAY)
    t3 = np.stack([g, g, g], axis = 2)
    return t3


class RGBTCC_RGBDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        img_size = (768, 1024),
        sigma = 15.0,
        max_count = None,
        return_pts = False,
        out_stride = 8,
    ):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)
        self.return_pts = bool(return_pts)
        self.out_stride = int(out_stride)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        rgb_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_RGB"), [".jpg", ".png"])
        if rgb_p is None:
            raise FileNotFoundError(f"Missing RGB for {sid} in {self.split_dir}")

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        bgr = cv2.imread(rgb_p)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        H0, W0 = rgb.shape[:2]
        rgb_res = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / float(W0))
            pts[:, 1] *= (self.h / float(H0))

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= self.out_stride
            pts_out[:, 1] /= self.out_stride

        den = density_from_points(
            pts_out,
            self.h_out,
            self.w_out,
            sigma = max(1.0, self.sigma / self.out_stride),
        )

        gt_count = float(len(pts))
        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_res).contiguous()
        den_t = _den_to_tensor(den)

        if self.return_pts:
            pts_out_t = torch.from_numpy(pts_out.astype(np.float32, copy = False)).contiguous()
            return rgb_t, den_t, pts_out_t, f"{sid}.jpg", gt_count

        return rgb_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_TDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        img_size = (768, 1024),
        sigma = 15.0,
        max_count = None,
        return_pts = False,
        out_stride = 8,
    ):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)
        self.return_pts = bool(return_pts)
        self.out_stride = int(out_stride)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_T.jpg") or f.endswith("_T.png")]
        ids = sorted({n.replace("_T.jpg", "").replace("_T.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_T images in {self.split_dir}")
        self.ids = ids

        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        t_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_T"), [".jpg", ".png"])
        if t_p is None:
            raise FileNotFoundError(f"Missing T for {sid} in {self.split_dir}")

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        t_raw = cv2.imread(t_p)
        if t_raw is None:
            raise FileNotFoundError(f"Cannot read: {t_p}")
        t3 = _to_t3(t_raw)

        H0, W0 = t3.shape[:2]
        t3_r = cv2.resize(t3, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / float(W0))
            pts[:, 1] *= (self.h / float(H0))

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= self.out_stride
            pts_out[:, 1] /= self.out_stride

        den = density_from_points(
            pts_out,
            self.h_out,
            self.w_out,
            sigma = max(1.0, self.sigma / self.out_stride),
        )

        gt_count = float(len(pts))
        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        t3_t = _tf_t3(t3_r).contiguous()
        den_t = _den_to_tensor(den)

        if self.return_pts:
            pts_out_t = torch.from_numpy(pts_out.astype(np.float32, copy = False)).contiguous()
            return t3_t, den_t, pts_out_t, f"{sid}.jpg", gt_count

        return t3_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_PairedDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        img_size = (768, 1024),
        sigma = 15.0,
        max_count = None,
        return_pts = False,
        out_stride = 8,
    ):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)
        self.return_pts = bool(return_pts)
        self.out_stride = int(out_stride)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        rgb_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_RGB"), [".jpg", ".png"])
        t_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_T"), [".jpg", ".png"])
        if rgb_p is None or t_p is None:
            raise FileNotFoundError(f"Missing RGB/T for {sid} in {self.split_dir}")

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        rgb_bgr = cv2.imread(rgb_p)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        t_raw = cv2.imread(t_p)
        if t_raw is None:
            raise FileNotFoundError(f"Cannot read: {t_p}")
        t3 = _to_t3(t_raw)

        H0, W0 = rgb.shape[:2]
        rgb_r = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t3_r = cv2.resize(t3, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / float(W0))
            pts[:, 1] *= (self.h / float(H0))

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= self.out_stride
            pts_out[:, 1] /= self.out_stride

        den = density_from_points(
            pts_out,
            self.h_out,
            self.w_out,
            sigma = max(1.0, self.sigma / self.out_stride),
        )

        gt_count = float(len(pts))
        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_r).contiguous()
        t3_t = _tf_t3(t3_r).contiguous()
        den_t = _den_to_tensor(den)

        if self.return_pts:
            pts_out_t = torch.from_numpy(pts_out.astype(np.float32, copy = False)).contiguous()
            return rgb_t, t3_t, den_t, pts_out_t, f"{sid}.jpg", gt_count

        return rgb_t, t3_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_EarlyFusionDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        img_size = (768, 1024),
        sigma = 15.0,
        max_count = None,
        return_pts = False,
        out_stride = 8,
    ):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)
        self.return_pts = bool(return_pts)
        self.out_stride = int(out_stride)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        rgb_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_RGB"), [".jpg", ".png"])
        t_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_T"), [".jpg", ".png"])
        if rgb_p is None or t_p is None:
            raise FileNotFoundError(f"Missing RGB/T for {sid} in {self.split_dir}")

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        rgb_bgr = cv2.imread(rgb_p)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        t1 = cv2.imread(t_p, cv2.IMREAD_GRAYSCALE)
        if t1 is None:
            raise FileNotFoundError(f"Cannot read: {t_p}")

        H0, W0 = rgb.shape[:2]
        rgb_r = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t1_r = cv2.resize(t1, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / float(W0))
            pts[:, 1] *= (self.h / float(H0))

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= self.out_stride
            pts_out[:, 1] /= self.out_stride

        den = density_from_points(
            pts_out,
            self.h_out,
            self.w_out,
            sigma = max(1.0, self.sigma / self.out_stride),
        )

        gt_count = float(len(pts))
        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_r).contiguous()
        t1_t = _tf_t1(t1_r)[0:1, :, :].contiguous()
        x4 = torch.cat([rgb_t, t1_t], dim = 0).contiguous()

        den_t = _den_to_tensor(den)

        if self.return_pts:
            pts_out_t = torch.from_numpy(pts_out.astype(np.float32, copy = False)).contiguous()
            return x4, den_t, pts_out_t, f"{sid}.jpg", gt_count

        return x4, den_t, f"{sid}.jpg", gt_count


# -------------------------------------------------------------------------
# Backwards-compatible API (for older scripts that import RGBTCCDset / RGBTCCBase)
# -------------------------------------------------------------------------
from dataclasses import dataclass
from typing import List, Sequence, Union


@dataclass(frozen=True)
class RGBTCCBase:
    rgb_path: str
    t_path: str
    gt_path: str
    image_id: str


def _first_existing_dir(base, candidates):
    from pathlib import Path
    base = Path(base)
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    return None


def _match_by_stem(dir_path, stem, exts):
    from pathlib import Path
    d = Path(dir_path)
    for e in exts:
        p = d / f"{stem}{e}"
        if p.exists():
            return p
    matches = sorted(d.glob(f"{stem}.*"))
    return matches[0] if matches else None


def _read_id_list(txt_path):
    ids = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s.rsplit(".", 1)[0])
    return ids


def build_splits_rgbt_cc(data_root: str, split_root: str = ""):
    # Returns base_train, base_val as List[RGBTCCBase].
    from pathlib import Path

    data_root_p = Path(data_root)
    split_root_p = Path(split_root) if split_root else None

    def build_one(split_name: str) -> List[RGBTCCBase]:
        split_dir = data_root_p / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        rgb_dir = _first_existing_dir(split_dir, ["RGB", "rgb", "images_rgb", "Images/RGB", "images/RGB"])
        t_dir = _first_existing_dir(split_dir, ["T", "t", "thermal", "Thermal", "images_t", "Images/T", "images/T"])
        gt_dir = _first_existing_dir(split_dir, ["GT", "gt", "annotations", "ann", "label", "Labels", "Images/GT", "images/GT"])

        flat_layout = False
        if rgb_dir is None or t_dir is None or gt_dir is None:
            # Flat layout: files live directly under the split directory with names like
            #   <id>_RGB.jpg, <id>_T.jpg, <id>_GT.json
            flat_layout = True
            rgb_dir = split_dir
            t_dir = split_dir
            gt_dir = split_dir

        id_list = None
        if split_root_p is not None:
            cand = [f"{split_name}.txt", f"{split_name}_list.txt", f"{split_name}list.txt"]
            for c in cand:
                p = split_root_p / c
                if p.exists():
                    id_list = _read_id_list(p)
                    break

        img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        gt_exts = [".json", ".npy", ".npz", ".mat"]

        if id_list is None:
            rgb_paths = []
            if flat_layout:
                # Only use RGB images as anchors (avoid T / GT files).
                for e in img_exts:
                    rgb_paths.extend(split_dir.glob(f"*_RGB{e}"))
            else:
                for e in img_exts:
                    rgb_paths.extend(rgb_dir.glob(f"*{e}"))

            stems = []
            for p in sorted(rgb_paths):
                st = p.stem
                if st.endswith("_RGB"):
                    st = st[:-4]
                stems.append(st)
            stems = sorted(set(stems))
        else:
            stems = id_list

        base = []
        for stem in stems:
            if flat_layout:
                rgb_p = _match_by_stem(rgb_dir, f"{stem}_RGB", img_exts) or _match_by_stem(rgb_dir, stem, img_exts)
                t_p = _match_by_stem(t_dir, f"{stem}_T", img_exts) or _match_by_stem(t_dir, stem, img_exts)
                gt_p = _match_by_stem(gt_dir, f"{stem}_GT", gt_exts) or _match_by_stem(gt_dir, stem, gt_exts)
            else:
                rgb_p = _match_by_stem(rgb_dir, stem, img_exts) or _match_by_stem(rgb_dir, f"{stem}_RGB", img_exts)
                t_p = _match_by_stem(t_dir, stem, img_exts) or _match_by_stem(t_dir, f"{stem}_T", img_exts)
                gt_p = _match_by_stem(gt_dir, stem, gt_exts) or _match_by_stem(gt_dir, f"{stem}_GT", gt_exts)
            if rgb_p is None or t_p is None or gt_p is None:
                continue
            base.append(RGBTCCBase(str(rgb_p), str(t_p), str(gt_p), stem))

        if len(base) == 0:
            raise RuntimeError(f"No valid (RGB,T,GT) triplets found for split='{split_name}' under {split_dir}")
        return base

    base_train = build_one("train")
    try:
        base_val = build_one("val")
    except FileNotFoundError:
        base_val = build_one("test")
    return base_train, base_val


def _pad_to_min_size(img, min_h: int, min_w: int):
    import numpy as np
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img
    if img.ndim == 2:
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)


def _paired_random_crop_params(h: int, w: int, crop_h: int, crop_w: int, deterministic: bool):
    if h == crop_h:
        top = 0
    elif deterministic:
        top = (h - crop_h) // 2
    else:
        top = random.randint(0, h - crop_h)

    if w == crop_w:
        left = 0
    elif deterministic:
        left = (w - crop_w) // 2
    else:
        left = random.randint(0, w - crop_w)
    return top, left


def _apply_crop_hflip_rgb_t_pts(rgb_np, t_np, pts_xy, crop_h: int, crop_w: int, deterministic: bool, is_train: bool):
    import numpy as np

    rgb_np = _pad_to_min_size(rgb_np, crop_h, crop_w)
    t_np = _pad_to_min_size(t_np, crop_h, crop_w)

    h, w = rgb_np.shape[:2]
    top, left = _paired_random_crop_params(h, w, crop_h, crop_w, deterministic)

    rgb_c = rgb_np[top:top + crop_h, left:left + crop_w]
    t_c = t_np[top:top + crop_h, left:left + crop_w]

    pts = pts_xy.copy().astype(np.float32)
    if pts.size > 0:
        inside = (
            (pts[:, 0] >= left) & (pts[:, 0] < left + crop_w) &
            (pts[:, 1] >= top) & (pts[:, 1] < top + crop_h)
        )
        pts = pts[inside]
        pts[:, 0] -= left
        pts[:, 1] -= top

    do_flip = False
    if is_train and not deterministic:
        do_flip = random.random() < 0.5

    if do_flip:
        rgb_c = np.ascontiguousarray(rgb_c[:, ::-1, :])
        t_c = np.ascontiguousarray(t_c[:, ::-1] if t_c.ndim == 2 else t_c[:, ::-1, :])
        if pts.size > 0:
            pts[:, 0] = (crop_w - 1) - pts[:, 0]

    return rgb_c, t_c, pts


class RGBTCCDset(torch.utils.data.Dataset):
    # Paired RGB-T dataset returning density maps (CSRNet-style).
    def __init__(
        self,
        base: Sequence[Union[RGBTCCBase, dict]],
        crop_size: int = 224,
        sigma: float = 15.0,
        down: int = 8,
        is_train: bool = True,
        deterministic: bool = False,
    ):
        self.base = list(base)
        self.crop_h = int(crop_size)
        self.crop_w = int(crop_size)
        self.sigma = float(sigma)
        self.down = int(down)
        self.is_train = bool(is_train)
        self.deterministic = bool(deterministic)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        import numpy as np
        from pathlib import Path
        from PIL import Image

        item = self.base[idx]
        if isinstance(item, dict):
            rgb_path = item["rgb_path"]
            t_path = item["t_path"]
            gt_path = item["gt_path"]
            image_id = item.get("image_id", Path(rgb_path).stem)
        else:
            rgb_path = item.rgb_path
            t_path = item.t_path
            gt_path = item.gt_path
            image_id = item.image_id

        rgb_img = Image.open(rgb_path).convert("RGB")
        t_img = Image.open(t_path).convert("L")

        rgb_np = np.asarray(rgb_img)
        t_np = np.asarray(t_img)

        pts = load_points(gt_path)
        if pts is None:
            pts = np.zeros((0, 2), dtype=np.float32)

        rgb_np, t_np, pts_crop = _apply_crop_hflip_rgb_t_pts(
            rgb_np, t_np, pts_xy=pts, crop_h=self.crop_h, crop_w=self.crop_w,
            deterministic=self.deterministic, is_train=self.is_train
        )

        rgb_t = _normalize_imagenet(_to_chw_tensor_rgb(rgb_np))

        t_1 = _to_chw_tensor_gray01(t_np)
        t_3 = t_1.repeat(3, 1, 1)
        t_t = _normalize_imagenet(t_3)

        out_h = self.crop_h // self.down
        out_w = self.crop_w // self.down

        if pts_crop.size == 0:
            den = np.zeros((out_h, out_w), dtype=np.float32)
        else:
            pts_ds = pts_crop / float(self.down)
            # Use the local CSRNet-style density builder (count-preserving normalization)
            den = density_from_points(pts_ds, out_h, out_w, sigma=self.sigma / float(self.down))

        den_t = torch.from_numpy(den).unsqueeze(0).float()

        meta = {
            "rgb_path": rgb_path,
            "t_path": t_path,
            "gt_path": gt_path,
            "count": int(pts_crop.shape[0]),
            "crop_size": (self.crop_h, self.crop_w),
            "down": self.down,
        }
        return rgb_t, t_t, den_t, image_id, meta


class RGBTCC_RGBDset(torch.utils.data.Dataset):
    def __init__(self, base, crop_size: int = 224, sigma: float = 15.0, down: int = 8, is_train: bool = True, deterministic: bool = False):
        self.paired = RGBTCCDset(base, crop_size=crop_size, sigma=sigma, down=down, is_train=is_train, deterministic=deterministic)

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx: int):
        rgb_t, _t_t, den_t, image_id, meta = self.paired[idx]
        return rgb_t, den_t, image_id, meta


class RGBTCC_TDset(torch.utils.data.Dataset):
    def __init__(self, base, crop_size: int = 224, sigma: float = 15.0, down: int = 8, is_train: bool = True, deterministic: bool = False):
        self.paired = RGBTCCDset(base, crop_size=crop_size, sigma=sigma, down=down, is_train=is_train, deterministic=deterministic)

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx: int):
        _rgb_t, t_t, den_t, image_id, meta = self.paired[idx]
        return t_t, den_t, image_id, meta
