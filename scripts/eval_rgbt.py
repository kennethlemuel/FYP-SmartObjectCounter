import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from datasets.rgbt_cc import RGBTCC_PairedDataset, RGBTCC_EarlyFusionDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location = device)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    return ckpt


def _game(pred_den, gt_den, L):
    h, w = pred_den.shape
    k = 2 ** L
    hs = max(1, h // k)
    ws = max(1, w // k)

    err = 0.0
    for i in range(k):
        for j in range(k):
            y0 = i * hs
            y1 = (i + 1) * hs if i < k - 1 else h
            x0 = j * ws
            x1 = (j + 1) * ws if j < k - 1 else w
            p = float(pred_den[y0:y1, x0:x1].sum().item())
            g = float(gt_den[y0:y1, x0:x1].sum().item())
            err += abs(p - g)
    return err


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["early", "late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split", default = "val", choices = ["train", "val", "test"])
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--num_workers", type = int, default = 2)
    args = ap.parse_args()

    device = get_device()

    if args.mode == "early":
        ds = RGBTCC_EarlyFusionDataset(
            root = args.data_root,
            split = args.split,
            img_size = (args.img_h, args.img_w),
            sigma = args.sigma,
        )
        model = CSRNetRGBT_Early(load_imagenet = True).to(device)
    else:
        ds = RGBTCC_PairedDataset(
            root = args.data_root,
            split = args.split,
            img_size = (args.img_h, args.img_w),
            sigma = args.sigma,
        )
        model = CSRNetRGBT_Late(load_imagenet = True).to(device)

    dl = DataLoader(
        ds,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers if device.type == "cuda" else 0,
        pin_memory = (device.type == "cuda"),
        drop_last = False,
    )

    sd = _load_state_dict(args.ckpt, device)
    model.load_state_dict(sd, strict = True)
    model.eval()

    mae = 0.0
    rmse = 0.0
    game = [0.0, 0.0, 0.0, 0.0]

    for batch in dl:
        if args.mode == "early":
            x4, den, _, _ = batch
            x4 = x4.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x4)
        else:
            x_rgb, x_t3, den, _, _ = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x_rgb, x_t3)

        if pred.shape[-2:] != den.shape[-2:]:
            den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)

        c_pred = float(pred.sum().item())
        c_gt = float(den.sum().item())

        mae += abs(c_pred - c_gt)
        rmse += (c_pred - c_gt) ** 2

        p2 = pred[0, 0]
        g2 = den[0, 0]
        for L in range(4):
            game[L] += _game(p2, g2, L)

    n = len(dl)
    mae /= n
    rmse = math.sqrt(rmse / n)
    game = [g / n for g in game]

    print(f"[rgbt-{args.mode}] split = {args.split}  MAE = {mae:.2f}  RMSE = {rmse:.2f}  GAME0 = {game[0]:.2f}  GAME1 = {game[1]:.2f}  GAME2 = {game[2]:.2f}  GAME3 = {game[3]:.2f}")


if __name__ == "__main__":
    main()