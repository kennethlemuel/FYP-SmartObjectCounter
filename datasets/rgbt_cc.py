# datasets/rgbt_cc.py
import os, json
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
    s = dm.sum()
    if s > 0:
        dm *= (len(xs) / s)
    return dm

def _load_points_json(p):
    with open(p, "r") as f:
        data = json.load(f)
    for k in ["points", "keypoints", "annotations"]:
        if k in data and isinstance(data[k], list):
            pts = np.array(data[k], dtype = np.float32).reshape(-1, 2)
            return pts
    if "labels" in data and isinstance(data["labels"], list):
        pts = np.array(data["labels"], dtype = np.float32).reshape(-1, 2)
        return pts
    return np.zeros((0, 2), dtype = np.float32)

def _load_points_mat(p):
    m = loadmat(p)
    if "point" in m:
        pts = np.array(m["point"], dtype = np.float32).reshape(-1, 2)
        return pts
    if "image_info" in m:
        pts = m["image_info"][0, 0][0, 0][0]
        return np.array(pts, dtype = np.float32).reshape(-1, 2)
    return np.zeros((0, 2), dtype = np.float32)

def load_points(label_path_no_ext):
    json_p = label_path_no_ext + ".json"
    mat_p  = label_path_no_ext + ".mat"
    if os.path.exists(json_p):
        return _load_points_json(json_p)
    if os.path.exists(mat_p):
        return _load_points_mat(mat_p)
    return np.zeros((0, 2), dtype = np.float32)

_imagenet_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])

def _to_t3(bgr_img):
    # read T image (grayscale or 1ch) and replicate to 3 channels
    if len(bgr_img.shape) == 2:
        g = bgr_img
    else:
        g = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    t3 = np.stack([g, g, g], axis = 2)
    return t3

# ---------- Datasets ----------

class RGBTCC_RGBDataset(Dataset):
    """ RGB-only baseline """
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.h, self.w = img_size
        self.sigma = sigma
        self.split_dir = os.path.join(root, split)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted(list(set([n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names])))
        if max_count is not None: ids = ids[:max_count]
        if len(ids) == 0: raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids
        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE
        self.tf = _imagenet_tf

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.jpg")
        if not os.path.exists(rgb_p):
            rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.png")
        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        img_bgr = cv2.imread(rgb_p); img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]
        img_res = cv2.resize(img_rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / W0); pts[:, 1] *= (self.h / H0)

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= OUT_STRIDE; pts_out[:, 1] /= OUT_STRIDE
        den = density_from_points(pts_out, self.h_out, self.w_out, sigma = max(1.0, self.sigma / OUT_STRIDE))

        return self.tf(img_res), torch.from_numpy(den).unsqueeze(0), f"{sid}.jpg", float(len(pts))

class RGBTCC_TDataset(Dataset):
    """ Thermal-only baseline (T replicated to 3ch, same CSRNet) """
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.root = root; self.split = split
        self.h, self.w = img_size; self.sigma = sigma
        self.split_dir = os.path.join(root, split)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_T.jpg") or f.endswith("_T.png")]
        ids = sorted(list(set([n.replace("_T.jpg", "").replace("_T.png", "") for n in names])))
        if max_count is not None: ids = ids[:max_count]
        if len(ids) == 0: raise RuntimeError(f"No *_T images in {self.split_dir}")
        self.ids = ids
        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE
        self.tf = _imagenet_tf

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        t_p = os.path.join(self.split_dir, f"{sid}_T.jpg")
        if not os.path.exists(t_p):
            t_p = os.path.join(self.split_dir, f"{sid}_T.png")
        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        img_bgr = cv2.imread(t_p); t3 = _to_t3(img_bgr)          # replicate to 3ch
        H0, W0 = t3.shape[:2]
        img_res = cv2.resize(t3, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / W0); pts[:, 1] *= (self.h / H0)

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= OUT_STRIDE; pts_out[:, 1] /= OUT_STRIDE
        den = density_from_points(pts_out, self.h_out, self.w_out, sigma = max(1.0, self.sigma / OUT_STRIDE))

        return _imagenet_tf(img_res), torch.from_numpy(den).unsqueeze(0), f"{sid}.jpg", float(len(pts))

class RGBTCC_PairedDataset(Dataset):
    """ Returns both RGB(3ch) and T(3ch) tensors for fusion baselines """
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.root = root; self.split = split
        self.h, self.w = img_size; self.sigma = sigma
        self.split_dir = os.path.join(root, split)

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted(list(set([n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names])))
        if max_count is not None: ids = ids[:max_count]
        if len(ids) == 0: raise RuntimeError(f"No *_RGB images in {self.split_dir}")
        self.ids = ids
        self.h_out, self.w_out = self.h // OUT_STRIDE, self.w // OUT_STRIDE
        self.tf = _imagenet_tf

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.jpg")
        if not os.path.exists(rgb_p):
            rgb_p = os.path.join(self.split_dir, f"{sid}_RGB.png")
        t_p = os.path.join(self.split_dir, f"{sid}_T.jpg")
        if not os.path.exists(t_p):
            t_p = os.path.join(self.split_dir, f"{sid}_T.png")
        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")

        rgb_bgr = cv2.imread(rgb_p); rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        t_bgr = cv2.imread(t_p); t3 = _to_t3(t_bgr)

        H0, W0 = rgb.shape[:2]
        rgb_r = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t3_r  = cv2.resize(t3,  (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / W0); pts[:, 1] *= (self.h / H0)

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= OUT_STRIDE; pts_out[:, 1] /= OUT_STRIDE
        den = density_from_points(pts_out, self.h_out, self.w_out, sigma = max(1.0, self.sigma / OUT_STRIDE))

        rgb_t = self.tf(rgb_r)             # [3,H,W]
        t3_t  = self.tf(t3_r)              # [3,H,W]
        return rgb_t, t3_t, torch.from_numpy(den).unsqueeze(0), f"{sid}.jpg", float(len(pts))
    
    # --- add below your existing imports and helpers in datasets/rgbt_cc.py ---

class RGBTCC_RGBTDataset(Dataset):
    """
    RGB-T loader for RGBT-CC (CVPR'21).
    Expects files like:  ####_RGB.jpg/.png,  ####_T.jpg/.png,  ####_GT.json/.mat
    Returns: rgb_tensor (3xHxW), t_tensor (1xHxW), den_map (1xhxw), name, gt_count
    """
    def __init__(self, root, split, img_size = (768, 1024), sigma = 15.0, max_count = None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.h, self.w = img_size
        self.sigma = sigma

        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = sorted({n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names})
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images found in {self.split_dir}")
        self.ids = ids

        self.tf_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
        self.tf_t = transforms.Compose([
            transforms.ToTensor(),                      # HxWx1 in [0,1]
            transforms.Normalize(mean = [0.5], std = [0.5]),
        ])

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

        # read RGB
        bgr = cv2.imread(rgb_p)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {rgb_p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # read thermal (grayscale)
        t_img = cv2.imread(t_p, cv2.IMREAD_GRAYSCALE)
        if t_img is None:
            raise FileNotFoundError(f"Cannot read: {t_p}")

        H0, W0 = rgb.shape[:2]
        rgb_res = cv2.resize(rgb, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        t_res   = cv2.resize(t_img, (self.w, self.h), interpolation = cv2.INTER_LINEAR)

        # load and scale GT points
        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")
        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / float(W0))
            pts[:, 1] *= (self.h / float(H0))

        # downsampled coordinates for density
        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= OUT_STRIDE
            pts_out[:, 1] /= OUT_STRIDE

        sigma_out = max(1.0, self.sigma / OUT_STRIDE)
        den = density_from_points(pts_out, self.h_out, self.w_out, sigma = sigma_out)

        gt_count = float(len(pts))
        s = den.sum()
        if gt_count > 0 and s > 0:
            den *= (gt_count / s)

        rgb_t = self.tf_rgb(rgb_res)
        t_t   = self.tf_t(t_res)[0:1, :, :]   # ensure shape (1,H,W)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return rgb_t, t_t, den_t, f"{sid}.jpg", gt_count