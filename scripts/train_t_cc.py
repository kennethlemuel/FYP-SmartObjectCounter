import argparse
import os
import random
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.csrnet import CSRNet
from datasets.rgbt_cc import RGBTCC


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(0) == 1:
        return x.repeat(3, 1, 1)
    if x.dim() == 3 and x.size(0) == 3:
        return x
    raise ValueError(f"Unexpected thermal tensor shape: {tuple(x.shape)}")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    abs_err = []
    sq_err = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            rgb, t, gt = batch[0], batch[1], batch[-1]
        elif isinstance(batch, dict):
            rgb = batch.get("rgb", None)
            t = batch.get("t", None) or batch.get("thermal", None)
            gt = batch.get("gt", None) or batch.get("density", None)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if t is None or gt is None:
            raise RuntimeError("Dataset batch must provide thermal (t/thermal) and gt (gt/density).")

        t = t.to(device, non_blocking = True).float()
        gt = gt.to(device, non_blocking = True).float()

        if t.dim() == 3:
            t = t.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        t3 = torch.stack([to_3ch(t[i]) for i in range(t.size(0))], dim = 0)

        pred = model(t3)
        pred_cnt = pred.sum(dim = (1, 2, 3))
        gt_cnt = gt.sum(dim = (1, 2, 3))

        e = (pred_cnt - gt_cnt).abs().detach().cpu().numpy()
        abs_err.extend(e.tolist())
        sq_err.extend((e ** 2).tolist())

    mae = float(np.mean(abs_err)) if abs_err else 0.0
    rmse = float(math.sqrt(np.mean(sq_err))) if sq_err else 0.0
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type = str, required = True)
    parser.add_argument("--img_h", type = int, default = 768)
    parser.add_argument("--img_w", type = int, default = 1024)
    parser.add_argument("--sigma", type = float, default = 15.0)

    parser.add_argument("--epochs", type = int, default = 30)
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--lr", type = float, default = 1e-5)
    parser.add_argument("--weight_decay", type = float, default = 1e-4)
    parser.add_argument("--num_workers", type = int, default = 2)
    parser.add_argument("--amp", action = "store_true")

    parser.add_argument("--save_dir", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 42)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = RGBTCC(
        data_root = args.data_root,
        split = "train",
        img_h = args.img_h,
        img_w = args.img_w,
        sigma = args.sigma,
    )
    val_set = RGBTCC(
        data_root = args.data_root,
        split = "val",
        img_h = args.img_h,
        img_w = args.img_w,
        sigma = args.sigma,
    )

    train_loader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size = 1,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = False,
    )

    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction = "mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled = args.amp)

    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                rgb, t_img, gt = batch[0], batch[1], batch[-1]
            elif isinstance(batch, dict):
                t_img = batch.get("t", None) or batch.get("thermal", None)
                gt = batch.get("gt", None) or batch.get("density", None)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            if t_img is None or gt is None:
                raise RuntimeError("Dataset batch must provide thermal (t/thermal) and gt (gt/density).")

            t_img = t_img.to(device, non_blocking = True).float()
            gt = gt.to(device, non_blocking = True).float()

            if t_img.dim() == 3:
                t_img = t_img.unsqueeze(0)
            if gt.dim() == 3:
                gt = gt.unsqueeze(0)

            t3 = torch.stack([to_3ch(t_img[i]) for i in range(t_img.size(0))], dim = 0)

            optimizer.zero_grad(set_to_none = True)
            with torch.cuda.amp.autocast(enabled = args.amp):
                pred = model(t3)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())

        val_mae, val_rmse = evaluate(model, val_loader, device)

        last_path = os.path.join(args.save_dir, "last_t.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if val_mae < best_mae:
            best_mae = val_mae
            best_path = os.path.join(args.save_dir, "best_t.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mae": val_mae, "val_rmse": val_rmse}, best_path)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={running_loss / max(len(train_loader), 1):.6f} "
              f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} time={dt:.1f}s")


if __name__ == "__main__":
    main()