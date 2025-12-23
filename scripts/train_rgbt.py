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
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate
from datasets.rgbt_cc import RGBTCC_RGBDataset, RGBTCC_TDataset, RGBTCC_PairedDataset


def set_seed(s = 42, deterministic = True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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


class EMA:
    def __init__(self, model, decay = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow.keys():
            self.shadow[k].mul_(self.decay).add_(msd[k], alpha = 1.0 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict = True)


@torch.no_grad()
def evaluate(model, loader, device, mode):
    model.eval()
    mae = 0.0
    rmse = 0.0

    for batch in loader:
        if mode == "rgb":
            x_rgb, den, _, _ = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x_rgb)

        elif mode == "t":
            x_t1, den, _, _ = batch
            x_t1 = x_t1.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            x_t3 = x_t1.repeat(1, 3, 1, 1)
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
    ap.add_argument("--mode", choices = ["rgb", "t", "early", "late", "adaptive_late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split_train", default = "train")
    ap.add_argument("--split_val", default = "val")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--lr", type = float, default = 5e-6)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--num_workers", type = int, default = -1)
    ap.add_argument("--seed", type = int, default = 42)

    ap.add_argument("--clip_grad", type = float, default = 0.5)
    ap.add_argument("--warmup_epochs", type = int, default = 2)
    ap.add_argument("--ema_decay", type = float, default = 0.999)
    ap.add_argument("--plateau_patience", type = int, default = 3)
    ap.add_argument("--plateau_factor", type = float, default = 0.5)
    ap.add_argument("--min_lr", type = float, default = 1e-7)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(args.seed, deterministic = True)

    device = get_device()
    print(f"[init] device = {device}")

    img_size = (args.img_h, args.img_w)

    if args.mode == "rgb":
        train_ds = RGBTCC_RGBDataset(args.data_root, args.split_train, img_size, args.sigma)
        val_ds = RGBTCC_RGBDataset(args.data_root, args.split_val, img_size, args.sigma)
    elif args.mode == "t":
        train_ds = RGBTCC_TDataset(args.data_root, args.split_train, img_size, args.sigma)
        val_ds = RGBTCC_TDataset(args.data_root, args.split_val, img_size, args.sigma)
    else:
        train_ds = RGBTCC_PairedDataset(args.data_root, args.split_train, img_size, args.sigma)
        val_ds = RGBTCC_PairedDataset(args.data_root, args.split_val, img_size, args.sigma)

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

    if args.mode == "rgb":
        model = CSRNet(load_imagenet = True).to(device)
    elif args.mode == "t":
        model = CSRNet(load_imagenet = True).to(device)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = True).to(device)
    elif args.mode == "late":
        model = CSRNetRGBT_Late(load_imagenet = True).to(device)
    else:
        model = CSRNetRGBT_AdaptiveLate(load_imagenet = True).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    mse = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled = (args.amp and device.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode = "min",
        factor = args.plateau_factor,
        patience = args.plateau_patience,
        min_lr = args.min_lr,
        verbose = False,
    )

    ema = EMA(model, decay = args.ema_decay) if args.ema_decay > 0 else None

    best_mae = float("inf")
    base_lr = args.lr

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            lr_w = base_lr * (ep / float(args.warmup_epochs))
            for pg in optim.param_groups:
                pg["lr"] = lr_w

        for step, batch in enumerate(train_dl, 1):
            optim.zero_grad(set_to_none = True)

            with torch.amp.autocast(
                device_type = _autocast_device_type(device),
                enabled = (args.amp and device.type == "cuda"),
            ):
                if args.mode == "rgb":
                    x_rgb, den, _, _ = batch
                    x_rgb = x_rgb.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    pred = model(x_rgb)

                elif args.mode == "t":
                    x_t1, den, _, _ = batch
                    x_t1 = x_t1.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    x_t3 = x_t1.repeat(1, 3, 1, 1)
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

            if args.clip_grad is not None and args.clip_grad > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)

            scaler.step(optim)
            scaler.update()

            if ema is not None:
                ema.update(model)

            run_loss += float(loss.item())

            if step == 1 or step % 50 == 0:
                print(f"[e{ep:02d} s{step:04d}/{len(train_dl)}] loss = {loss.item():.4f}")

        train_loss = run_loss / max(1, len(train_dl))

        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
            val_mae, val_rmse = evaluate(model, val_dl, device, args.mode)
            model.load_state_dict(backup, strict = True)
        else:
            val_mae, val_rmse = evaluate(model, val_dl, device, args.mode)

        scheduler.step(val_mae)

        last_path = os.path.join(args.save_dir, f"{args.mode}_last.pth")
        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "lr": optim.param_groups[0]["lr"],
            },
            last_path
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_path = os.path.join(args.save_dir, f"{args.mode}_best.pth")
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "lr": optim.param_groups[0]["lr"],
                },
                best_path
            )
            print(f"-> saved {os.path.basename(best_path)} (MAE {val_mae:.2f})")

        dt = time.time() - t0
        print(
            f"Epoch {ep:02d}: train_loss = {train_loss:.4f}  MAE = {val_mae:.2f}  RMSE = {val_rmse:.2f}  "
            f"lr = {optim.param_groups[0]['lr']:.2e}  time = {dt:.1f}s"
        )


if __name__ == "__main__":
    main()