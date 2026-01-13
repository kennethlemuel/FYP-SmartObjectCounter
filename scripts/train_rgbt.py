import os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#Models
from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate

#Dataset
from datasets.rgbt_cc import RGBTCCDset, RGBTCCBase, build_splits_rgbt_cc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #Determinism knobs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    #Ensure each worker has deterministic, distinct seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _autocast_device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _resize_density_sum_preserving(d: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Resizes the density map (B,1,H,W) to (B,1,h,w) while preserving total sum (count).
    Uses 'area' (avg) interpolation then scales by area ratio.
    """
    if d.dim() != 4:
        raise ValueError(f"density must be 4D (B,1,H,W), got {tuple(d.shape)}")
    H, W = d.shape[-2:]
    h, w = size_hw
    if (H, W) == (h, w):
        return d
    d_rs = F.interpolate(d, size = (h, w), mode = "area")
    scale = (H * W) / float(h * w)
    return d_rs * scale


@torch.no_grad()
def evaluate(model: nn.Module, val_dl: DataLoader, device: torch.device, mode: str) -> Tuple[float, float, Dict[int, float]]:
    """
    Returns: MAE (GAME0), RMSE, GAME{1..3} dict
    """
    model.eval()

    abs_errs = []
    sq_errs = []

    # GAME metrics
    game_abs = {1: [], 2: [], 3: []}

    for batch in val_dl:
        if mode == "rgb":
            x_rgb, den, meta, _ = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x_rgb)

        elif mode == "t":
            x_t, den, meta, _ = batch
            x_t = x_t.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            if x_t.shape[1] == 1:
                x_t = x_t.repeat(1, 3, 1, 1)
            pred = model(x_t)

        elif mode == "early":
            x4, den, meta, _ = batch
            x4 = x4.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x4)

        else:
            x_rgb, x_t3, den, meta, _ = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            if x_t3.shape[1] == 1:
                x_t3 = x_t3.repeat(1, 3, 1, 1)
            pred = model(x_rgb, x_t3)

        if pred.shape[-2:] != den.shape[-2:]:
            den = _resize_density_sum_preserving(den, pred.shape[-2:])

        pred_cnt = float(pred.sum().item())
        gt_cnt = float(den.sum().item())

        err = pred_cnt - gt_cnt
        abs_errs.append(abs(err))
        sq_errs.append(err * err)

        # GAME (grid-based MAE)
        for L in [1, 2, 3]:
            g = 2**L
            # split into g x g
            ph, pw = pred.shape[-2], pred.shape[-1]
            cell_h = ph // g
            cell_w = pw // g
            # crop to multiple
            ph2 = cell_h * g
            pw2 = cell_w * g
            p = pred[..., :ph2, :pw2]
            d = den[..., :ph2, :pw2]
            p_cells = p.reshape(1, 1, g, cell_h, g, cell_w).sum(dim = (3, 5))
            d_cells = d.reshape(1, 1, g, cell_h, g, cell_w).sum(dim = (3, 5))
            game_abs[L].append(float(torch.abs(p_cells - d_cells).sum().item()))

    mae = float(np.mean(abs_errs)) if abs_errs else float("nan")
    rmse = float(np.sqrt(np.mean(sq_errs))) if sq_errs else float("nan")
    game = {L: float(np.mean(game_abs[L])) if game_abs[L] else float("nan") for L in [1, 2, 3]}
    return mae, rmse, game


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# =========================
# Main
# =========================

def parse_args():
    p = argparse.ArgumentParser("RGB-T Crowd Counting Training")

    # Data
    p.add_argument("--data_root", type = str, required = True)
    p.add_argument("--split_root", type = str, default = "")
    p.add_argument("--crop_size", type = int, default = 224)
    p.add_argument("--sigma", type = float, default = 15.0)
    p.add_argument("--down", type = int, default = 8)

    # Train
    p.add_argument("--mode", type = str, default = "late", choices = ["rgb", "t", "early", "late", "adaptive_late"])
    p.add_argument("--epochs", type = int, default = 100)
    p.add_argument("--batch_size", type = int, default = 1)
    p.add_argument("--lr", type = float, default = 1e-5)
    p.add_argument("--weight_decay", type = float, default = 1e-4)
    p.add_argument("--optimizer", type = str, default = "adam", choices = ["adam", "adamw"])
    p.add_argument("--amp", action = "store_true")
    p.add_argument("--grad_accum", type = int, default = 1)
    p.add_argument("--clip_grad", type = float, default = 0.0)

    # Adaptive-late extras
    p.add_argument("--gate_lr", type = float, default = 1e-4)
    p.add_argument("--freeze_backbones_epochs", type = int, default = 0)

    # Scheduler (optional)
    p.add_argument("--scheduler", type = str, default = "none", choices = ["none", "multistep", "onecycle"])
    p.add_argument("--milestones", type = str, default = "200,300")
    p.add_argument("--gamma", type = float, default = 0.1)
    p.add_argument("--max_lr", type = float, default = 2e-5)
    p.add_argument("--max_gate_lr", type = float, default = 2e-4)
    p.add_argument("--pct_start", type = float, default = 0.1)
    p.add_argument("--div_factor", type = float, default = 25.0)
    p.add_argument("--final_div_factor", type = float, default = 10000.0)

    # System
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--workers", type = int, default = 4)
    p.add_argument("--save_dir", type = str, default = "./ckpt")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    #Build split lists
    base_train, base_val = build_splits_rgbt_cc(args.data_root, args.split_root)

    #paired multimodal augmentation is inside RGBTCCDset for training
    #deterministic (no random crop/flip) for val.
    train_ds = RGBTCCDset(
        base = base_train,
        crop_size = args.crop_size,
        sigma = args.sigma,
        down = args.down,
        is_train = True,
    )
    val_ds = RGBTCCDset(
        base = base_val,
        crop_size = args.crop_size,
        sigma = args.sigma,
        down = args.down,
        is_train = False,   #deterministic center crop
    )

    pin = (device.type == "cuda")
    workers = args.workers if device.type == "cuda" else 0

    #Make dataloader deterministic across epochs via generator
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

    if args.mode == "adaptive_late":
        backbone_params = list(model.rgb_net.parameters()) + list(model.t_net.parameters())
        gate_params = list(model.gate.parameters())

        if args.optimizer == "adam":
            optim = torch.optim.Adam(
                [
                    {"params": backbone_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": gate_params, "lr": args.gate_lr, "weight_decay": args.weight_decay},
                ]
            )
        else:
            optim = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": gate_params, "lr": args.gate_lr, "weight_decay": args.weight_decay},
                ]
            )
    else:
        if args.optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        else:
            optim = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled = (args.amp and device.type == "cuda"))

    # Scheduler setup
    scheduler = None
    step_scheduler_per_optim_step = False

    steps_per_epoch_optim = int(math.ceil(len(train_dl) / float(max(1, args.grad_accum))))

    if args.scheduler == "multistep":
        ms = [int(x.strip()) for x in args.milestones.split(",") if x.strip()]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = ms, gamma = args.gamma)
        step_scheduler_per_optim_step = False  # epoch-level

    elif args.scheduler == "onecycle":
        if args.mode == "adaptive_late":
            max_lrs = [args.max_lr, args.max_gate_lr]
        else:
            max_lrs = args.max_lr

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr = max_lrs,
            epochs = args.epochs,
            steps_per_epoch = steps_per_epoch_optim,
            pct_start = args.pct_start,
            div_factor = args.div_factor,
            final_div_factor = args.final_div_factor,
            anneal_strategy = "cos",
        )
        step_scheduler_per_optim_step = True  # batch-level (but only when optim.step happens)

    best_mae = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()

        if args.mode == "adaptive_late" and args.freeze_backbones_epochs > 0:
            freeze = (ep <= args.freeze_backbones_epochs)
            _set_requires_grad(model.rgb_net, not freeze)
            _set_requires_grad(model.t_net, not freeze)
            _set_requires_grad(model.gate, True)

        run_loss = 0.0
        t0 = time.time()

        optim.zero_grad(set_to_none = True)

        for step, batch in enumerate(train_dl, 1):
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
                    den = _resize_density_sum_preserving(den, pred.shape[-2:])

                loss = mse(pred, den) / max(1, args.grad_accum)

            scaler.scale(loss).backward()
            run_loss += float(loss.item()) * max(1, args.grad_accum)

            do_step = (step % max(1, args.grad_accum) == 0)
            if do_step:
                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)

                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none = True)

                if scheduler is not None and step_scheduler_per_optim_step:
                    scheduler.step()

            if step == 1 or step % 50 == 0:
                print(f"[e{ep:03d} s{step:04d}/{len(train_dl)}] loss = {run_loss / step:.6f}")

        #Handle remainder accumulation (if len(train_dl) not divisible by grad_accum)
        if len(train_dl) % max(1, args.grad_accum) != 0:
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none = True)

            if scheduler is not None and step_scheduler_per_optim_step:
                scheduler.step()

        train_loss = run_loss / max(1, len(train_dl))
        val_mae, val_rmse, val_game = evaluate(model, val_dl, device, args.mode)

        #Epoch-level scheduler step (MultiStepLR)
        if scheduler is not None and (not step_scheduler_per_optim_step):
            scheduler.step()

        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_game": val_game,
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
                    "val_game": val_game,
                    "lr": optim.param_groups[0]["lr"],
                },
                os.path.join(args.save_dir, f"{args.mode}_best.pth")
            )
            print(f"-> saved {args.mode}_best.pth (MAE/GAME0 {val_mae:.2f})")

        dt = time.time() - t0
        g1 = val_game.get(1, float("nan"))
        g2 = val_game.get(2, float("nan"))
        g3 = val_game.get(3, float("nan"))
        lr0 = optim.param_groups[0]["lr"]

        print(
            f"Epoch {ep:03d}: train_loss = {train_loss:.6f}  "
            f"MAE/GAME0 = {val_mae:.2f}  RMSE = {val_rmse:.2f}  "
            f"GAME1 = {g1:.2f}  GAME2 = {g2:.2f}  GAME3 = {g3:.2f}  "
            f"lr = {lr0:.2e}  time = {dt:.1f}s"
        )


if __name__ == "__main__":
    main()
