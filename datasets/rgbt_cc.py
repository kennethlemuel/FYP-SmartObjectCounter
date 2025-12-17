import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat

OUT_STRIDE = 8

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

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


def density_from_points(points_xy, h, w, sigma = 15.0):
    dm = np.zeros((h, w), dtype = np.float32)
    if points_xy.size == 0:
        return dm
    xs = np.clip(points_xy[:, 0].astype(int), 0, w - 1)
    ys = np.clip(points_xy[:, 1].astype(int), 0, h - 1)
    dm[ys, xs] = 1.0
    dm = gaussian_filter(dm, sigma = sigma, mode = "constant")
    s = dm.sum()
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


def _to_t3(img_any):
    if img_any.ndim == 2:
        g = img_any
    else:
        g = cv2.cvtColor(img_any, cv2.COLOR_BGR2GRAY)
    t3 = np.stack([g, g, g], axis = 2)
    return t3


class RGBTCC_RGBDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = sigma

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE

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
            pts_out[:, 0] /= OUT_STRIDE
            pts_out[:, 1] /= OUT_STRIDE

        den = density_from_points(
            pts_out, self.h_out, self.w_out,
            sigma = max(1.0, self.sigma / OUT_STRIDE)
        )

        gt_count = float(len(pts))
        s = den.sum()
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        return _tf_rgb(rgb_res), torch.from_numpy(den).unsqueeze(0), f"{sid}.jpg", gt_count


class RGBTCC_TDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = sigma

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_T.jpg") or f.endswith("_T.png")]
        ids = sorted({n.replace("_T.jpg", "").replace("_T.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_T images in {self.split_dir}")
        self.ids = ids

        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE

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
            pts_out[:, 0] /= OUT_STRIDE
            pts_out[:, 1] /= OUT_STRIDE

        den = density_from_points(
            pts_out, self.h_out, self.w_out,
            sigma = max(1.0, self.sigma / OUT_STRIDE)
        )

        gt_count = float(len(pts))
        s = den.sum()
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        t3_t = _tf_t3(t3_r)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return t3_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_PairedDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = sigma

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE

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
            pts_out[:, 0] /= OUT_STRIDE
            pts_out[:, 1] /= OUT_STRIDE

        den = density_from_points(
            pts_out, self.h_out, self.w_out,
            sigma = max(1.0, self.sigma / OUT_STRIDE)
        )

        gt_count = float(len(pts))
        s = den.sum()
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_r)
        t3_t = _tf_t3(t3_r)
        den_t = torch.from_numpy(den).unsqueeze(0)

        return rgb_t, t3_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_EarlyFusionDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = sigma

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE

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
            pts_out[:, 0] /= OUT_STRIDE
            pts_out[:, 1] /= OUT_STRIDE

        den = density_from_points(
            pts_out, self.h_out, self.w_out,
            sigma = max(1.0, self.sigma / OUT_STRIDE)
        )

        gt_count = float(len(pts))
        s = den.sum()
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_r)
        t1_t = _tf_t1(t1_r)[0:1, :, :]
        x4 = torch.cat([rgb_t, t1_t], dim = 0)

        return x4, torch.from_numpy(den).unsqueeze(0), f"{sid}.jpg", gt_count