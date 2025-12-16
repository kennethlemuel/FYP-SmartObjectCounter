import os
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from datasets.rgbt_cc import RGBTCC_TDataset, RGBTCC_PairedDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["t", "early", "late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split", choices = ["train", "val", "test"], default = "val")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--num_workers", type = int, default = 2)
    args = ap.parse_args()

    device = get_device()
    print(f"[init] device = {device}")

    if args.mode == "t":
        ds = RGBTCC_TDataset(args.data_root, args.split, (args.img_h, args.img_w), args.sigma)
    else:
        ds = RGBTCC_PairedDataset(args.data_root, args.split, (args.img_h, args.img_w), args.sigma)

    dl = DataLoader(ds, batch_size = 1, shuffle = False, num_workers = args.num_workers)

    if args.mode == "t":
        model = CSRNet(load_imagenet = False).to(device)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = False).to(device)
    else:
        model = CSRNetRGBT_Late(load_imagenet = False).to(device)

    ckpt = torch.load(args.ckpt, map_location = "cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict = True)
    model.eval()

    mae = 0.0
    rmse = 0.0

    for batch in dl:
        if args.mode == "t":
            x_t3, den, _, _ = batch
            x_t3 = x_t3.to(device)
            den = den.to(device)
            pred = model(x_t3)
        else:
            x_rgb, x_t3, den, _, _ = batch
            x_rgb = x_rgb.to(device)
            x_t3 = x_t3.to(device)
            den = den.to(device)

            if args.mode == "early":
                t_gray = x_t3[:, :1, :, :]
                x4 = torch.cat([x_rgb, t_gray], dim = 1)
                pred = model(x4)
            else:
                pred = model(x_rgb, x_t3)

        if pred.shape[-2:] != den.shape[-2:]:
            den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)

        c_pred = float(pred.sum().item())
        c_gt = float(den.sum().item())
        mae += abs(c_pred - c_gt)
        rmse += (c_pred - c_gt) ** 2

    mae /= len(dl)
    rmse = math.sqrt(rmse / len(dl))
    print(f"[{args.mode}] split = {args.split}  MAE = {mae:.2f}  RMSE = {rmse:.2f}")


if __name__ == "__main__":
    main()