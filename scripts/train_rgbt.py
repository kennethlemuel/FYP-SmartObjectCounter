import os
import sys
import time
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate

from datasets.rgbt_cc import (
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
)

OUT_STRIDE = 8


def set_seed(s = 42, deterministic = True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    wseed = torch.initial_seed() % (2**32)
    random.seed(wseed)
    np.random.seed(wseed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _autocast_device_type(device):
    return "cuda" if device.type == "cuda" else "cpu"


def _hflip(x):
    return torch.flip(x, dims = [-1])


def _make_resizable(x):
    return x.contiguous().clone()


def _crop_params(h, w, crop, stride = OUT_STRIDE):
    ch = min(crop, h)
    cw = min(crop, w)

    ch = max(stride, (ch // stride) * stride)
    cw = max(stride, (cw // stride) * stride)

    max_y0 = h - ch
    max_x0 = w - cw

    if max_y0 <= 0:
        y0 = 0
    else:
        y0 = random.randint(0, max_y0 // stride) * stride

    if max_x0 <= 0:
        x0 = 0
    else:
        x0 = random.randint(0, max_x0 // stride) * stride

    return y0, y0 + ch, x0, x0 + cw


class TrainAugment(Dataset):
    def __init__(self, base, mode, crop_size = 256, flip_prob = 0.5, stride = OUT_STRIDE):
        self.base = base
        self.mode = mode
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.stride = stride

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        do_flip = (random.random() < self.flip_prob)

        if self.mode == "rgb":
            x_rgb, den, a, b = sample
            _, h, w = x_rgb.shape
            y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

            y0d, y1d = y0 // self.stride, y1 // self.stride
            x0d, x1d = x0 // self.stride, x1 // self.stride

            x_rgb = x_rgb[:, y0:y1, x0:x1]
            den = den[:, y0d:y1d, x0d:x1d]

            if do_flip:
                x_rgb = _hflip(x_rgb)
                den = _hflip(den)

            return _make_resizable(x_rgb), _make_resizable(den), a, b

        if self.mode == "t":
            x_t, den, a, b = sample
            _, h, w = x_t.shape
            y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

            y0d, y1d = y0 // self.stride, y1 // self.stride
            x0d, x1d = x0 // self.stride, x1 // self.stride

            x_t = x_t[:, y0:y1, x0:x1]
            den = den[:, y0d:y1d, x0d:x1d]

            if do_flip:
                x_t = _hflip(x_t)
                den = _hflip(den)

            return _make_resizable(x_t), _make_resizable(den), a, b

        if self.mode == "early":
            x4, den, a, b = sample
            _, h, w = x4.shape
            y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

            y0d, y1d = y0 // self.stride, y1 // self.stride
            x0d, x1d = x0 // self.stride, x1 // self.stride

            x4 = x4[:, y0:y1, x0:x1]
            den = den[:, y0d:y1d, x0d:x1d]

            if do_flip:
                x4 = _hflip(x4)
                den = _hflip(den)

            return _make_resizable(x4), _make_resizable(den), a, b

        x_rgb, x_t3, den, a, b = sample
        _, h, w = x_rgb.shape
        y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

        y0d, y1d = y0 // self.stride, y1 // self.stride
        x0d, x1d = x0 // self.stride, x1 // self.stride

        x_rgb = x_rgb[:, y0:y1, x0:x1]
        x_t3 = x_t3[:, y0:y1, x0:x1]
        den = den[:, y0d:y1d, x0d:x1d]

        if do_flip:
            x_rgb = _hflip(x_rgb)
            x_t3 = _hflip(x_t3)
            den = _hflip(den)

        return _make_resizable(x_rgb), _make_resizable(x_t3), _make_resizable(den), a, b


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
            x_t, den, _, _ = batch
            x_t = x_t.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            if x_t.shape[1] == 1:
                x_t = x_t.repeat(1, 3, 1, 1)
            pred = model(x_t)

        elif mode == "early":
            x4, den, _, _ = batch
            x4 = x4.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x4)

        else:
            x_rgb, x_t3, den, _, _ = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            if x_t3.shape[1] == 1:
                x_t3 = x_t3.repeat(1, 3, 1, 1)
            pred = model(x_rgb, x_t3)

        if pred.shape[-2:] != den.shape[-2:]:
            den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)

        c_pred = float(pred.sum().item())
        c_gt = float(den.sum().item())
        mae += abs(c_pred - c_gt)
        rmse += (c_pred - c_gt) ** 2

    n = max(1, len(loader))
    mae /= n
    rmse = math.sqrt(rmse / n)
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

    ap.add_argument("--epochs", type = int, default = 300)
    ap.add_argument("--batch_size", type = int, default = 16)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 0.0)

    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--num_workers", type = int, default = -1)
    ap.add_argument("--seed", type = int, default = 42)

    ap.add_argument("--crop_size", type = int, default = 256)
    ap.add_argument("--flip_prob", type = float, default = 0.5)

    ap.add_argument("--clip_grad", type = float, default = 0.0)

    ap.add_argument("--optimizer", choices = ["adam", "adamw"], default = "adam")
    ap.add_argument("--milestones", type = str, default = "200,250")
    ap.add_argument("--gamma", type = float, default = 0.1)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(args.seed, deterministic = True)

    device = get_device()
    print(f"[init] device = {device}")

    img_size = (args.img_h, args.img_w)

    if args.mode == "rgb":
        base_train = RGBTCC_RGBDataset(args.data_root, args.split_train, img_size, args.sigma)
        base_val = RGBTCC_RGBDataset(args.data_root, args.split_val, img_size, args.sigma)
    elif args.mode == "t":
        base_train = RGBTCC_TDataset(args.data_root, args.split_train, img_size, args.sigma)
        base_val = RGBTCC_TDataset(args.data_root, args.split_val, img_size, args.sigma)
    elif args.mode == "early":
        base_train = RGBTCC_EarlyFusionDataset(args.data_root, args.split_train, img_size, args.sigma)
        base_val = RGBTCC_EarlyFusionDataset(args.data_root, args.split_val, img_size, args.sigma)
    else:
        base_train = RGBTCC_PairedDataset(args.data_root, args.split_train, img_size, args.sigma)
        base_val = RGBTCC_PairedDataset(args.data_root, args.split_val, img_size, args.sigma)

    train_ds = TrainAugment(base_train, mode = args.mode, crop_size = args.crop_size, flip_prob = args.flip_prob)
    val_ds = base_val

    if args.num_workers < 0:
        workers = 4 if device.type == "cuda" else 0
    else:
        workers = args.num_workers

    pin = (device.type == "cuda") and (workers > 0)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dl = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = workers,
        pin_memory = pin,
        drop_last = True,
        worker_init_fn = seed_worker,
        generator = g,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size = 1,
        shuffle = False,
        num_workers = workers if device.type == "cuda" else 0,
        pin_memory = (device.type == "cuda"),
        drop_last = False,
    )

    print(f"[init] train = {len(base_train)}  val = {len(base_val)}  workers = {workers}")

    if args.mode in ["rgb", "t"]:
        model = CSRNet(load_imagenet = True).to(device)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = True).to(device)
    elif args.mode == "late":
        model = CSRNetRGBT_Late(load_imagenet = True).to(device)
    else:
        model = CSRNetRGBT_AdaptiveLate(load_imagenet = True).to(device)

    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled = (args.amp and device.type == "cuda"))

    ms = [int(x.strip()) for x in args.milestones.split(",") if x.strip()]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = ms, gamma = args.gamma)

    best_mae = float("inf")

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
                if args.mode == "rgb":
                    x_rgb, den, _, _ = batch
                    x_rgb = x_rgb.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    pred = model(x_rgb)

                elif args.mode == "t":
                    x_t, den, _, _ = batch
                    x_t = x_t.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    if x_t.shape[1] == 1:
                        x_t = x_t.repeat(1, 3, 1, 1)
                    pred = model(x_t)

                elif args.mode == "early":
                    x4, den, _, _ = batch
                    x4 = x4.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    pred = model(x4)

                else:
                    x_rgb, x_t3, den, _, _ = batch
                    x_rgb = x_rgb.to(device, non_blocking = True)
                    x_t3 = x_t3.to(device, non_blocking = True)
                    den = den.to(device, non_blocking = True)
                    if x_t3.shape[1] == 1:
                        x_t3 = x_t3.repeat(1, 3, 1, 1)
                    pred = model(x_rgb, x_t3)

                if pred.shape[-2:] != den.shape[-2:]:
                    den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)

                loss = mse(pred, den)

            scaler.scale(loss).backward()

            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)

            scaler.step(optim)
            scaler.update()

            run_loss += float(loss.item())

            if step == 1 or step % 50 == 0:
                print(f"[e{ep:03d} s{step:04d}/{len(train_dl)}] loss = {loss.item():.6f}")

        train_loss = run_loss / max(1, len(train_dl))
        val_mae, val_rmse = evaluate(model, val_dl, device, args.mode)

        scheduler.step()

        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "lr": optim.param_groups[0]["lr"],
            },
            os.path.join(args.save_dir, f"{args.mode}_last.pth")
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "lr": optim.param_groups[0]["lr"],
                },
                os.path.join(args.save_dir, f"{args.mode}_best.pth")
            )
            print(f"-> saved {args.mode}_best.pth (MAE {val_mae:.2f})")

        dt = time.time() - t0
        print(
            f"Epoch {ep:03d}: train_loss = {train_loss:.6f}  MAE = {val_mae:.2f}  RMSE = {val_rmse:.2f}  "
            f"lr = {optim.param_groups[0]['lr']:.2e}  time = {dt:.1f}s"
        )


if __name__ == "__main__":
    main()