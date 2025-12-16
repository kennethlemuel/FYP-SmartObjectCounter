import os
import time
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from datasets.rgbt_cc import RGBTCC_TDataset, RGBTCC_PairedDataset


def set_seed(s = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def _autocast_device_type(device):
    return "cuda" if device.type == "cuda" else "cpu"


@torch.no_grad()
def evaluate(model, loader, device, mode):
    model.eval()
    mae = 0.0
    rmse = 0.0

    for batch in loader:
        if mode == "t":
            x_t3, den, _, _ = batch
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x_t3)
        else:
            x_rgb, x_t3, den, _, _ = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)

            if mode == "early":
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

    mae /= len(loader)
    rmse = math.sqrt(rmse / len(loader))
    return mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["t", "early", "late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split_train", default = "train")
    ap.add_argument("--split_val", default = "val")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--num_workers", type = int, default = -1)
    ap.add_argument("--seed", type = int, default = 42)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(args.seed)

    device = get_device()
    print(f"[init] device = {device}")

    if args.mode == "t":
        train_ds = RGBTCC_TDataset(args.data_root, args.split_train, (args.img_h, args.img_w), args.sigma)
        val_ds = RGBTCC_TDataset(args.data_root, args.split_val, (args.img_h, args.img_w), args.sigma)
    else:
        train_ds = RGBTCC_PairedDataset(args.data_root, args.split_train, (args.img_h, args.img_w), args.sigma)
        val_ds = RGBTCC_PairedDataset(args.data_root, args.split_val, (args.img_h, args.img_w), args.sigma)

    if args.num_workers < 0:
        workers = 2 if device.type == "cuda" else 0
    else:
        workers = args.num_workers

    pin = (device.type == "cuda") and (workers > 0)

    train_dl = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = workers,
        pin_memory = pin,
        drop_last = False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size = 1,
        shuffle = False,
        num_workers = workers,
        pin_memory = pin,
        drop_last = False,
    )

    print(f"[init] train = {len(train_ds)}  val = {len(val_ds)}  workers = {workers}")

    if args.mode == "t":
        model = CSRNet(load_imagenet = True).to(device)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = True).to(device)
    else:
        model = CSRNetRGBT_Late(load_imagenet = True).to(device)

    optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled = (args.amp and device.type == "cuda"))

    best_mae = 1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_dl, 1):
            optim.zero_grad(set_to_none = True)

            with torch.amp.autocast(
                device_type = _autocast_device_type(device),
                enabled = (args.amp and device.type == "cuda"),
            ):
                if args.mode == "t":
                    x_t3, den, _, _ = batch
                    x_t3 = x_t3.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    pred = model(x_t3)
                else:
                    x_rgb, x_t3, den, _, _ = batch
                    x_rgb = x_rgb.to(device, non_blocking = True)
                    x_t3 = x_t3.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)

                    if args.mode == "early":
                        t_gray = x_t3[:, :1, :, :]
                        x4 = torch.cat([x_rgb, t_gray], dim = 1)
                        pred = model(x4)
                    else:
                        pred = model(x_rgb, x_t3)

                if pred.shape[-2:] != den.shape[-2:]:
                    den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)

                loss = mse(pred, den)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            run_loss += float(loss.item())

            if step == 1 or step % 50 == 0:
                print(f"[e{ep:02d} s{step:04d}/{len(train_dl)}] loss = {loss.item():.4f}")

        train_loss = run_loss / max(1, len(train_dl))
        val_mae, val_rmse = evaluate(model, val_dl, device, args.mode)

        torch.save(
            {"epoch": ep, "model": model.state_dict()},
            os.path.join(args.save_dir, f"{args.mode}_last.pth")
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"epoch": ep, "model": model.state_dict(), "val_mae": val_mae, "val_rmse": val_rmse},
                os.path.join(args.save_dir, f"{args.mode}_best.pth")
            )

        dt = time.time() - t0
        print(f"Epoch {ep:02d}: train_loss = {train_loss:.4f}  MAE = {val_mae:.2f}  RMSE = {val_rmse:.2f}  time = {dt:.1f}s")


if __name__ == "__main__":
    main()