import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import re


def _parse_seq_frame(sid: str):
    m = re.match(r"^(.*?)(?:[_-])?(\d+)$", sid)
    if m is None:
        return sid, 0
    seq = m.group(1)
    seq = seq if len(seq) > 0 else sid
    frame = int(m.group(2))
    return seq, frame


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


def _den_to_tensor(den_hw: np.ndarray) -> torch.Tensor:
    den_hw = np.ascontiguousarray(den_hw.astype(np.float32, copy = False))
    return torch.from_numpy(den_hw).unsqueeze(0).contiguous()


def _resize_img(img: np.ndarray, w: int, h: int) -> np.ndarray:
    h0, w0 = img.shape[:2]
    if w0 == w and h0 == h:
        return img
    if w < w0 or h < h0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation = interp)


def _to_t3(img_any: np.ndarray) -> np.ndarray:
    if img_any.ndim == 2:
        g = img_any
    else:
        g = cv2.cvtColor(img_any, cv2.COLOR_BGR2GRAY)
    t3 = np.stack([g, g, g], axis = 2)
    return t3


def _read_thermal_any(path: str) -> np.ndarray:
    """
    Robust thermal read:
    - If uint16/float, convert to uint8 in a consistent way.
    - Return a single-channel uint8 image.
    """
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if im.dtype == np.uint8:
        return im

    # Typical 16-bit thermal: scale by dtype range (not per-image minmax)
    if im.dtype == np.uint16:
        im_f = im.astype(np.float32) / 65535.0
        im_u8 = (im_f * 255.0).clip(0.0, 255.0).astype(np.uint8)
        return im_u8

    # Fallback for float/other int types: minmax to [0,255]
    im_f = im.astype(np.float32)
    mn = float(im_f.min())
    mx = float(im_f.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(im_f, dtype = np.uint8)
    im_u8 = ((im_f - mn) / (mx - mn) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return im_u8


def density_from_points(points_xy: np.ndarray, h: int, w: int, sigma: float = 15.0) -> np.ndarray:
    """
    points_xy: (N,2) in (x,y) coordinates in the SAME space as (h,w).
    Returns a density map normalized so sum == N (when N > 0).

    Fix: accumulate collisions correctly via np.add.at.
    """
    dm = np.zeros((h, w), dtype = np.float32)
    if points_xy.size == 0:
        return dm

    pts = points_xy.astype(np.float32, copy = False)
    xs = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)

    # Important: preserve multiplicity when multiple points fall on same pixel
    np.add.at(dm, (ys, xs), 1.0)

    sig = float(sigma)
    if sig > 0.0:
        dm = gaussian_filter(dm, sigma = sig, mode = "constant")

    s = float(dm.sum())
    n = float(pts.shape[0])
    if n > 0.0 and s > 0.0:
        dm *= (n / s)
    return dm


def _load_points_json(p: str) -> np.ndarray:
    with open(p, "r") as f:
        data = json.load(f)

    for k in ["points", "keypoints", "annotations", "labels"]:
        if k not in data or not isinstance(data[k], list):
            continue
        arr = data[k]
        if len(arr) == 0:
            return np.zeros((0, 2), dtype = np.float32)

        # Support list of dicts: [{"x":..,"y":..}, ...]
        if isinstance(arr[0], dict):
            xs = []
            ys = []
            for it in arr:
                if not isinstance(it, dict):
                    continue
                if "x" in it and "y" in it:
                    xs.append(float(it["x"]))
                    ys.append(float(it["y"]))
                elif "X" in it and "Y" in it:
                    xs.append(float(it["X"]))
                    ys.append(float(it["Y"]))
            if len(xs) == 0:
                return np.zeros((0, 2), dtype = np.float32)
            return np.stack([np.array(xs, dtype = np.float32), np.array(ys, dtype = np.float32)], axis = 1)

        # Default: flat list or list of pairs
        pts = np.array(arr, dtype = np.float32).reshape(-1, 2)
        return pts

    return np.zeros((0, 2), dtype = np.float32)


def _load_points_mat(p: str) -> np.ndarray:
    m = loadmat(p)

    if "point" in m:
        pts = np.array(m["point"], dtype = np.float32)
        return pts.reshape(-1, 2)

    if "image_info" in m:
        pts = m["image_info"][0, 0][0, 0][0]
        return np.array(pts, dtype = np.float32).reshape(-1, 2)

    return np.zeros((0, 2), dtype = np.float32)


def load_points(label_path_no_ext: str) -> np.ndarray:
    json_p = label_path_no_ext + ".json"
    mat_p = label_path_no_ext + ".mat"

    if os.path.exists(json_p):
        return _load_points_json(json_p)
    if os.path.exists(mat_p):
        return _load_points_mat(mat_p)

    return np.zeros((0, 2), dtype = np.float32)


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

        bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        H0, W0 = rgb.shape[:2]
        rgb_res = _resize_img(rgb, self.w, self.h)

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

        gt_count = float(pts.shape[0])
        s = float(den.sum())
        if gt_count > 0.0 and s > 0.0:
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
        prefer_rgb_size_for_gt = True,
    ):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)
        self.return_pts = bool(return_pts)
        self.out_stride = int(out_stride)
        self.prefer_rgb_size_for_gt = bool(prefer_rgb_size_for_gt)

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

        # Thermal read (robust to 16-bit)
        t1 = _read_thermal_any(t_p)
        Ht, Wt = t1.shape[:2]
        t1_r = _resize_img(t1, self.w, self.h)
        t3_r = np.stack([t1_r, t1_r, t1_r], axis = 2)

        # Prefer RGB original size if GT coordinates are defined on RGB (common)
        H0, W0 = Ht, Wt
        if self.prefer_rgb_size_for_gt:
            rgb_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_RGB"), [".jpg", ".png"])
            if rgb_p is not None:
                rgb_bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
                if rgb_bgr is not None:
                    Hr, Wr = rgb_bgr.shape[:2]
                    H0, W0 = Hr, Wr

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

        gt_count = float(pts.shape[0])
        s = float(den.sum())
        if gt_count > 0.0 and s > 0.0:
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
        return_meta = False,
        temporal_pair = False,
        pair_delta = 1,
    ):
        assert split in ["train", "val", "test"]
        self.split_dir = os.path.join(root, split)
        self.h, self.w = img_size
        self.sigma = float(sigma)
        self.return_pts = bool(return_pts)
        self.out_stride = int(out_stride)
        self.return_meta = bool(return_meta)
        self.temporal_pair = bool(temporal_pair)
        self.pair_delta = max(1, int(pair_delta))

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids

        self.h_out = self.h // self.out_stride
        self.w_out = self.w // self.out_stride

        self._pair_index = None
        if self.temporal_pair:
            if not self.return_pts:
                raise ValueError("temporal_pair = True requires return_pts = True.")

            seq_to = {}
            for i, sid in enumerate(self.ids):
                seq, frame = _parse_seq_frame(sid)
                seq_to.setdefault(seq, []).append((frame, i))

            for seq in seq_to:
                seq_to[seq] = [i for _, i in sorted(seq_to[seq], key = lambda x: x[0])]

            pair_index = {}
            for seq, idxs in seq_to.items():
                n = len(idxs)
                for pos, i in enumerate(idxs):
                    j_pos = pos + self.pair_delta
                    if j_pos >= n:
                        j_pos = pos - self.pair_delta
                    if j_pos < 0 or j_pos >= n:
                        j_pos = pos
                    pair_index[i] = idxs[j_pos]
            self._pair_index = pair_index

    def __len__(self):
        return len(self.ids)

    def _get_one(self, idx):
        sid = self.ids[idx]

        rgb_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_RGB"), [".jpg", ".png"])
        t_p = _pick_existing(os.path.join(self.split_dir, f"{sid}_T"), [".jpg", ".png"])
        if rgb_p is None or t_p is None:
            raise FileNotFoundError(f"Missing RGB/T for {sid} in {self.split_dir}")

        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        rgb_bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        t1 = _read_thermal_any(t_p)
        t3 = np.stack([t1, t1, t1], axis = 2)

        H0, W0 = rgb.shape[:2]
        rgb_r = _resize_img(rgb, self.w, self.h)
        t3_r = _resize_img(t3, self.w, self.h)

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

        gt_count = float(pts.shape[0])
        s = float(den.sum())
        if gt_count > 0.0 and s > 0.0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_r).contiguous()
        t3_t = _tf_t3(t3_r).contiguous()
        den_t = _den_to_tensor(den)

        name = f"{sid}.jpg"
        seq, frame = _parse_seq_frame(sid)

        if self.return_pts:
            pts_out_t = torch.from_numpy(pts_out.astype(np.float32, copy = False)).contiguous()
            if self.return_meta:
                return rgb_t, t3_t, den_t, pts_out_t, name, gt_count, seq, frame
            return rgb_t, t3_t, den_t, pts_out_t, name, gt_count

        if self.return_meta:
            return rgb_t, t3_t, den_t, name, gt_count, seq, frame
        return rgb_t, t3_t, den_t, name, gt_count

    def __getitem__(self, idx):
        if not self.temporal_pair:
            return self._get_one(idx)

        j = self._pair_index.get(idx, idx)
        a = self._get_one(idx)
        b = self._get_one(j)
        return a, b


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

        rgb_bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        t1 = _read_thermal_any(t_p)

        H0, W0 = rgb.shape[:2]
        rgb_r = _resize_img(rgb, self.w, self.h)
        t1_r = _resize_img(t1, self.w, self.h)

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

        gt_count = float(pts.shape[0])
        s = float(den.sum())
        if gt_count > 0.0 and s > 0.0:
            den *= (gt_count / s)

        rgb_t = _tf_rgb(rgb_r).contiguous()
        t1_t = _tf_t1(t1_r)[0:1, :, :].contiguous()
        x4 = torch.cat([rgb_t, t1_t], dim = 0).contiguous()

        den_t = _den_to_tensor(den)

        if self.return_pts:
            pts_out_t = torch.from_numpy(pts_out.astype(np.float32, copy = False)).contiguous()
            return x4, den_t, pts_out_t, f"{sid}.jpg", gt_count

        return x4, den_t, f"{sid}.jpg", gt_count
