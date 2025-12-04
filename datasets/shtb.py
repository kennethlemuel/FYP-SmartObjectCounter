from typing import Tuple, List
import os, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
from .density import density_from_points_knn, density_from_points_fixed

class SHTBDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        img_size: Tuple[int, int] = (768, 1024),
        sigma: float = 15.0,
        max_count: int | None = None,
        out_stride: int = 8,
        crop_train: bool = True,
        strip_prefix: str = "processed_",
        use_adaptive_sigma: bool = True,
    ):
        assert split in {"train", "val"}
        self.root = root
        self.split = split
        self.h, self.w = img_size
        self.sigma = sigma
        self.out_stride = out_stride
        self.crop_train = crop_train and (split == "train")
        self.strip_prefix = strip_prefix
        self.use_adaptive_sigma = use_adaptive_sigma

        self.img_dir = os.path.join(root, "images")
        self.gt_dir = os.path.join(root, "ground_truth")

        imgs = sorted(
            [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        train_list = [f for i, f in enumerate(imgs) if i % 5 != 0]
        val_list = [f for i, f in enumerate(imgs) if i % 5 == 0]
        self.files = train_list if split == "train" else val_list
        if max_count is not None:
            self.files = self.files[: max_count]

        self.tf_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.crop_h = min(self.h, 640)
        self.crop_w = min(self.w, 640)

        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

    def __len__(self) -> int:
        return len(self.files)

    def _gt_path_from_image_name(self, name_no_ext: str) -> str:
        base = name_no_ext
        if self.strip_prefix and base.startswith(self.strip_prefix):
            base = base[len(self.strip_prefix) :]
        return os.path.join(self.gt_dir, f"GT_{base}.mat")

    @staticmethod
    def _load_points_from_mat(path: str) -> np.ndarray:
        mat = loadmat(path)
        if "image_info" in mat:
            pts = mat["image_info"][0, 0][0, 0][0]
            return np.array(pts, dtype=np.float32)
        if "annPoints" in mat:
            pts = mat["annPoints"]
            return np.array(pts, dtype=np.float32)
        raise KeyError(f"Unknown point key(s) in {path}")

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # resize to training size
        img_res = cv2.resize(img_rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # load gt points
        base_no_ext = os.path.splitext(name)[0]
        gt_path = self._gt_path_from_image_name(base_no_ext)
        pts = self._load_points_from_mat(gt_path)

        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / W)
            pts[:, 1] *= (self.h / H)

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

            img_res = img_res[y0:y1, x0:x1]
            if pts.size > 0:
                m = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
                pts = pts[m]
                if pts.size > 0:
                    pts[:, 0] -= x0
                    pts[:, 1] -= y0

            Ht, Wt = self.crop_h, self.crop_w
        else:
            Ht, Wt = self.h, self.w

        ho, wo = Ht // self.out_stride, Wt // self.out_stride
        if pts.size > 0:
            pts_out = pts.copy()
            pts_out[:, 0] *= (wo / float(Wt))
            pts_out[:, 1] *= (ho / float(Ht))
        else:
            pts_out = pts

        if self.use_adaptive_sigma:
            den = density_from_points_knn(pts_out, ho, wo, k=3, beta=0.3, normalize_to_count=True)
        else:
            den = density_from_points_fixed(pts_out, ho, wo, sigma=max(1.0, self.sigma / self.out_stride), normalize_to_count=True)

        gt_count = float(len(pts))

        img_t = self.tf_rgb(img_res)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return img_t, den_t, name, gt_count