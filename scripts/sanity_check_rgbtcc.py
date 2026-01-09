import argparse
import random
import numpy as np
import torch

from datasets.rgbt_cc import (
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
    density_from_points,
)

# Copy of the crop logic from your train script (keep identical)
def crop_params(h: int, w: int, crop: int, stride: int):
    ch = min(crop, h)
    cw = min(crop, w)

    ch = max(stride, (ch // stride) * stride)
    cw = max(stride, (cw // stride) * stride)

    y0 = 0 if h == ch else random.randint(0, h - ch)
    x0 = 0 if w == cw else random.randint(0, w - cw)

    y1 = y0 + ch
    x1 = x0 + cw
    return y0, y1, x0, x1

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def count_pts_in_crop(pts_out: torch.Tensor, y0: int, y1: int, x0: int, x1: int, stride: int) -> int:
    oy0 = int(y0 // stride)
    oy1 = int(y1 // stride)
    ox0 = int(x0 // stride)
    ox1 = int(x1 // stride)

    pts = pts_out.cpu().numpy()
    if pts.size == 0:
        return 0
    m = (pts[:, 0] >= ox0) & (pts[:, 0] < ox1) & (pts[:, 1] >= oy0) & (pts[:, 1] < oy1)
    return int(m.sum())

def build_crop_density_from_points(pts_out, y0, y1, x0, x1, out_stride, sigma, h, w):
    oy0 = int(y0 // out_stride)
    oy1 = int(y1 // out_stride)
    ox0 = int(x0 // out_stride)
    ox1 = int(x1 // out_stride)

    h_out = max(1, oy1 - oy0)
    w_out = max(1, ox1 - ox0)

    if pts_out.numel() == 0:
        return torch.zeros((1, h_out, w_out), dtype = torch.float32)

    pts = pts_out.clone()
    m = (pts[:, 0] >= ox0) & (pts[:, 0] < ox1) & (pts[:, 1] >= oy0) & (pts[:, 1] < oy1)
    pts_c = pts[m]
    if pts_c.numel() == 0:
        return torch.zeros((1, h_out, w_out), dtype = torch.float32)

    pts_c[:, 0] -= float(ox0)
    pts_c[:, 1] -= float(oy0)

    sigma_out = max(1.0, float(sigma) / float(out_stride))
    dm = density_from_points(pts_c.cpu().numpy(), h_out, w_out, sigma = sigma_out)

    gt_n = float(pts_c.shape[0])
    s = float(dm.sum())
    if gt_n > 0.0 and s > 0.0:
        dm = dm * (gt_n / s)

    return torch.from_numpy(dm.astype(np.float32, copy = False))[None, ...]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["rgb", "t", "early", "late", "adaptive_late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split", default = "train")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--out_stride", type = int, default = 8)
    ap.add_argument("--crop_size", type = int, default = 224)
    ap.add_argument("--n", type = int, default = 30)
    ap.add_argument("--seed", type = int, default = 42)
    args = ap.parse_args()

    set_seed(args.seed)
    img_size = (args.img_h, args.img_w)

    # return_pts=True is required for this sanity check
    if args.mode == "rgb":
        ds = RGBTCC_RGBDataset(args.data_root, args.split, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
    elif args.mode == "t":
        ds = RGBTCC_TDataset(args.data_root, args.split, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
    elif args.mode == "early":
        ds = RGBTCC_EarlyFusionDataset(args.data_root, args.split, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
    else:
        ds = RGBTCC_PairedDataset(args.data_root, args.split, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)

    eps_full = 1e-3
    eps_crop = 1e-2

    for k in range(args.n):
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]

        if args.mode in ["late", "adaptive_late"]:
            x_rgb, x_t3, den_full, pts_out, name, gt_count = sample
            _, h, w = x_rgb.shape
        elif args.mode == "early":
            x4, den_full, pts_out, name, gt_count = sample
            _, h, w = x4.shape
        else:
            x, den_full, pts_out, name, gt_count = sample
            _, h, w = x.shape

        full_sum = float(den_full.sum().item())
        n_pts_full = int(pts_out.shape[0])

        y0, y1, x0, x1 = crop_params(h, w, args.crop_size, stride = args.out_stride)

        den_crop = build_crop_density_from_points(
            pts_out = pts_out,
            y0 = y0, y1 = y1, x0 = x0, x1 = x1,
            out_stride = args.out_stride,
            sigma = args.sigma,
            h = h, w = w,
        )
        den_crop_sum = float(den_crop.sum().item())
        n_pts_crop = count_pts_in_crop(pts_out, y0, y1, x0, x1, stride = args.out_stride)

        ok_full = (abs(full_sum - gt_count) < eps_full) and (abs(full_sum - n_pts_full) < eps_full)
        ok_crop = abs(den_crop_sum - float(n_pts_crop)) < eps_crop

        status = "OK" if (ok_full and ok_crop) else "CHECK"
        print(
            f"[{k+1:02d}] {status} {name} | "
            f"gt_count = {gt_count:.0f}, pts = {n_pts_full}, den_full_sum = {full_sum:.3f} | "
            f"pts_in_crop = {n_pts_crop}, den_crop_sum = {den_crop_sum:.3f}"
        )

if __name__ == "__main__":
    main()
