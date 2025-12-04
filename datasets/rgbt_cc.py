import os, json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat

OUT_STRIDE = 8  #a CSRNet output downsample

def density_from_points(points_xy, h, w, sigma=15.0):
    """Build a Gaussian density map on an (h,w) grid from (N,2) point coords."""
    dm = np.zeros((h, w), dtype=np.float32)
    if points_xy.size == 0:
        return dm
    xs = np.clip(points_xy[:, 0].astype(int), 0, w - 1)
    ys = np.clip(points_xy[:, 1].astype(int), 0, h - 1)
    dm[ys, xs] = 1.0
    dm = gaussian_filter(dm, sigma=sigma, mode="constant")
    s = dm.sum()
    if s > 0:
        dm *= (len(xs) / s)
    return dm

def _load_points_json(p):
    with open(p, "r") as f:
        data = json.load(f)
    for k in ["points", "keypoints", "annotations"]:
        if k in data and isinstance(data[k], list):
            pts = np.array(data[k], dtype=np.float32)
            if pts.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
            pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
            return pts
    #a fallback (if stored under diff shape)
    if "labels" in data and isinstance(data["labels"], list):
        pts = np.array(data["labels"], dtype=np.float32).reshape(-1, 2)
        return pts
    return np.zeros((0, 2), dtype=np.float32)

def _load_points_mat(p):
    m = loadmat(p)
    if "point" in m:
        pts = np.array(m["point"], dtype=np.float32)
        return pts.reshape(-1, 2)
    if "image_info" in m:
        pts = m["image_info"][0, 0][0, 0][0]
        return np.array(pts, dtype=np.float32).reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)

def load_points(label_path_no_ext):
    """Try JSON first (native RGBT-CC), then MAT, else empty."""
    json_p = label_path_no_ext + ".json"
    mat_p  = label_path_no_ext + ".mat"
    if os.path.exists(json_p):
        return _load_points_json(json_p)
    if os.path.exists(mat_p):
        return _load_points_mat(mat_p)
    return np.zeros((0, 2), dtype=np.float32)

class RGBTCC_RGBDataset(Dataset):
    """
    RGB baseline loader for RGBT-CC (CVPR'21).
    Expects: data/RGBT-CC-CVPR2021/{train,val,test}/####_RGB.jpg, ####_T.jpg, ####_GT.json
    """
    def __init__(self, root, split, img_size=(768, 1024), sigma=15.0, max_count=None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.h, self.w = img_size
        self.sigma = sigma
        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")
        names = [f for f in os.listdir(self.split_dir) if f.endswith("_RGB.jpg") or f.endswith("_RGB.png")]
        ids = [n.replace("_RGB.jpg", "").replace("_RGB.png", "") for n in names]
        ids = sorted(list(set(ids)))
        if max_count is not None:
            ids = ids[:max_count]
        if len(ids) == 0:
            raise RuntimeError(f"No *_RGB images found in {self.split_dir}")
        self.ids = ids

        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),])
        self.h_out = self.h // OUT_STRIDE
        self.w_out = self.w // OUT_STRIDE

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb_path = os.path.join(self.split_dir, f"{sid}_RGB.jpg")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.split_dir, f"{sid}_RGB.png")
        gt_no_ext = os.path.join(self.split_dir, f"{sid}_GT")
        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {rgb_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        img_res = cv2.resize(img_rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        pts = load_points(gt_no_ext)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] *= (self.w / W)
            pts[:, 1] *= (self.h / H)

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:, 0] /= OUT_STRIDE
            pts_out[:, 1] /= OUT_STRIDE

        sigma_out = max(1.0, self.sigma / OUT_STRIDE)
        den = density_from_points(pts_out, self.h_out, self.w_out, sigma=sigma_out)

        img_t = self.tf(img_res)
        den_t = torch.from_numpy(den).unsqueeze(0)
        return img_t, den_t, f"{sid}.jpg", float(len(pts))