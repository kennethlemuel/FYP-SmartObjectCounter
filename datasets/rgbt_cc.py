
from __future__ import annotations
from typing import Tuple, List
import os, json, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .density import density_from_points_knn


def _first_existing(base: str, exts: List[str]) -> str | None:
    for e in exts:
        p = f"{base}{e}"
        if os.path.isfile(p):
            return p
    return None


class RGBTCCDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        img_size: Tuple[int, int] = (720, 960),
        sigma: float = 12.0,
        max_count: int | None = None,
        out_stride: int = 8,
        crop_train: bool = True,
        fuse: str = "early",  # "early" returns a 4-channel tensor; "dual" could return (rgb_t, t_t)
        t_norm: Tuple[float, float] = (0.5, 0.5),  # mean,std for thermal channel normalization
    ):
        assert split in {"train", "val", "test"}
        assert fuse in {"early"}
        self.root = root
        self.split = split
        self.h, self.w = img_size
        self.sigma = sigma
        self.out_stride = out_stride
        self.crop_train = crop_train and (split == "train")
        self.fuse = fuse
        self.t_mean, self.t_std = t_norm

        self.rgb_dir = os.path.join(root, "RGB")
        self.t_dir = os.path.join(root, "T")
        self.ann_dir = os.path.join(root, "annotations")
        self.splits_dir = os.path.join(root, "splits")

        split_file = os.path.join(self.splits_dir, f"{split}.txt")
        if os.path.isfile(split_file):
            with open(split_file, "r") as f:
                ids = [ln.strip() for ln in f if ln.strip()]
        else:
            rgb_ids = {os.path.splitext(f)[0] for f in os.listdir(self.rgb_dir)}
            t_ids = {os.path.splitext(f)[0] for f in os.listdir(self.t_dir)}
            ids = sorted(list(rgb_ids & t_ids))
            #naive split: 80/20
            if split != "train":
                ids = [i for k, i in enumerate(ids) if k % 5 == 0] if split == "val" else [i for k, i in enumerate(ids) if k % 7 == 0]
            else:
                ids = [i for k, i in enumerate(ids) if k % 5 != 0]

        self.ids = ids[: max_count] if max_count is not None else ids

        #resolved paths (we resolve extensions on-the-fly in __getitem__)
        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

        #crop window
        self.crop_h = min(self.h, 640)
        self.crop_w = min(self.w, 640)

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _load_points(path: str) -> np.ndarray:
        """
        Accept .mat with keys: 'points', 'annPoints', 'image_info'[…]
        Or .json: {"points": [[x,y], ...]}
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mat":
            from scipy.io import loadmat
            mat = loadmat(path)
            if "points" in mat:
                return np.array(mat["points"], dtype=np.float32)
            if "annPoints" in mat:
                return np.array(mat["annPoints"], dtype=np.float32)
            if "image_info" in mat:
                pts = mat["image_info"][0, 0][0, 0][0]
                return np.array(pts, dtype=np.float32)
            raise KeyError(f"Unknown keys in {path}")
        elif ext == ".json":
            with open(path, "r") as f:
                obj = json.load(f)
            pts = np.array(obj.get("points", []), dtype=np.float32)
            return pts
        else:
            raise ValueError(f"Unsupported annotation type: {ext}")

    def __getitem__(self, idx):
        stem = self.ids[idx]

        #resolve file paths (support .jpg/.png)
        rgb_path = _first_existing(os.path.join(self.rgb_dir, stem), [".jpg", ".png", ".jpeg"])
        t_path   = _first_existing(os.path.join(self.t_dir, stem),  [".jpg", ".png", ".jpeg"])
        ann_path = _first_existing(os.path.join(self.ann_dir, stem), [".mat", ".json"])
        if rgb_path is None or t_path is None or ann_path is None:
            raise FileNotFoundError(f"Missing files for id={stem} (rgb:{rgb_path}, t:{t_path}, ann:{ann_path})")

        #image loaders
        rgb_bgr = cv2.imread(rgb_path)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        t_gray = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
        if t_gray is None:
            raise FileNotFoundError(f"Cannot read Thermal: {t_path}")

        H, W = rgb.shape[:2]
        #resize both modalities (keep them aligned)
        rgb_res = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        t_res   = cv2.resize(t_gray, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        #loading & scale points
        pts = self._load_points(ann_path)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / W)
            pts[:, 1] *= (self.h / H)

        # Optional train crop (center near a point)
        if self.crop_train and (self.crop_h < self.h or self.crop_w < self.w):
            if pts.size > 0:
                j = np.random.randint(len(pts))
                cx = int(np.clip(pts[j, 0], self.crop_w // 2, self.w - self.crop_w // 2))
                cy = int(np.clip(pts[j, 1], self.crop_h // 2, self.h - self.crop_h // 2))
            else:
                cx = np.random.randint(self.crop_w // 2, self.w - self.crop_w // 2 + 1)
                cy = np.random.randint(self.crop_h // 2, self.h - self.crop_h // 2 + 1)

            x0 = cx - self.crop_w // 2
            y0 = cy - self.crop_h // 2
            x1 = x0 + self.crop_w
            y1 = y0 + self.crop_h

            rgb_res = rgb_res[y0:y1, x0:x1]
            t_res   = t_res[y0:y1, x0:x1]

            if pts.size > 0:
                m = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
                pts = pts[m]
                if pts.size > 0:
                    pts[:, 0] -= x0
                    pts[:, 1] -= y0

            Ht, Wt = self.crop_h, self.crop_w
        else:
            Ht, Wt = self.h, self.w

        #output resolution
        ho, wo = Ht // self.out_stride, Wt // self.out_stride

        #downscaling points to output grid
        if pts.size > 0:
            pts_out = pts.copy()
            pts_out[:, 0] *= (wo / float(Wt))
            pts_out[:, 1] *= (ho / float(Ht))
        else:
            pts_out = pts

        #applying adaptive density
        den = density_from_points_knn(pts_out, ho, wo, k=3, beta=0.3, normalize_to_count=True)
        gt_count = float(len(pts))

        #early fusion tensor: stack RGB (3ch) + Thermal (1ch)
        rgb_f = rgb_res.astype(np.float32) / 255.0
        t_f   = t_res.astype(np.float32) / 255.0
        img4  = np.dstack([rgb_f, t_f])  # H, W, 4

        #normalize: ImageNet on RGB; simple (mean,std) on thermal
        mean = np.array([0.485, 0.456, 0.406, self.t_mean], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225, self.t_std], dtype=np.float32)
        img4 = (img4 - mean) / std
        img4_t = torch.from_numpy(img4).permute(2, 0, 1).contiguous()  # [4,H,W]

        den_t = torch.from_numpy(den).unsqueeze(0)  # [1,ho,wo]
        return img4_t, den_t, stem, gt_count