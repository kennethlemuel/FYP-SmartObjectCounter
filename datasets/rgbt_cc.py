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


def density_from_points(points_xy, h, w, sigma = 15.0):
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


_tf_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])


def _ensure_read(img, path):
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def _to_t3_from_gray(t_gray):
    return np.stack([t_gray, t_gray, t_gray], axis = 2)


def _get_ids(split_dir, suffixes):
    names = []
    for f in os.listdir(split_dir):
        for s in suffixes:
            if f.endswith(s):
                names.append(f)
                break

    ids = []
    for n in names:
        for s in suffixes:
            if n.endswith(s):
                ids.append(n[:-len(s)])
                break
    ids = sorted(set(ids))
    if len(ids) == 0:
        raise RuntimeError(f"No matching files in {split_dir} for suffixes = {suffixes}")
    return ids


class RGBTCC_RGBDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)

        self.ids = _get_ids(self.split_dir, ["_RGB.jpg", "_RGB.png"])
        if max_count is not None:
            self.ids = self.ids[:max_count]

        self.h_out = self.h // OUT_STRIDE
        self.w_out = self.w // OUT_STRIDE

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.jpg")
        if not os.path.exists(rgb_p):
            rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.png")

        bgr = _ensure_read(cv2.imread(rgb_p), rgb_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]
        rgb_res = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")
        pts = load_points(gt_no_ext)
        gt_count = float(len(pts))

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

        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_imagenet(rgb_res)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return rgb_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_TDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)

        self.ids = _get_ids(self.split_dir, ["_T.jpg", "_T.png"])
        if max_count is not None:
            self.ids = self.ids[:max_count]

        self.h_out = self.h // OUT_STRIDE
        self.w_out = self.w // OUT_STRIDE

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        t_p = os.path.join(self.split_dir, f"{sid}_T.jpg")
        if not os.path.exists(t_p):
            t_p = os.path.join(self.split_dir, f"{sid}_T.png")

        t_gray = _ensure_read(cv2.imread(t_p, cv2.IMREAD_GRAYSCALE), t_p)
        H0, W0 = t_gray.shape[:2]

        t_res = cv2.resize(t_gray, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t3 = _to_t3_from_gray(t_res)

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")
        pts = load_points(gt_no_ext)
        gt_count = float(len(pts))

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

        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        t3_t = _tf_imagenet(t3)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return t3_t, den_t, f"{sid}.jpg", gt_count


class RGBTCC_PairedDataset(Dataset):
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)

        self.ids = _get_ids(self.split_dir, ["_RGB.jpg", "_RGB.png"])
        if max_count is not None:
            self.ids = self.ids[:max_count]

        self.h_out = self.h // OUT_STRIDE
        self.w_out = self.w // OUT_STRIDE

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.jpg")
        if not os.path.exists(rgb_p):
            rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.png")

        t_p = os.path.join(self.split_dir, f"{sid}_T.jpg")
        if not os.path.exists(t_p):
            t_p = os.path.join(self.split_dir, f"{sid}_T.png")

        rgb_bgr = _ensure_read(cv2.imread(rgb_p), rgb_p)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]

        t_gray = _ensure_read(cv2.imread(t_p, cv2.IMREAD_GRAYSCALE), t_p)

        rgb_res = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t_res = cv2.resize(t_gray, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t3 = _to_t3_from_gray(t_res)

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")
        pts = load_points(gt_no_ext)
        gt_count = float(len(pts))

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

        s = float(den.sum())
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = _tf_imagenet(rgb_res)
        t3_t = _tf_imagenet(t3)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return rgb_t, t3_t, den_t, f"{sid}.jpg", gt_count