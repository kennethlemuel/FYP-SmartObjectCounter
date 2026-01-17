import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 2.6 deprecates torch.cuda.amp.GradScaler
try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler

from datasets.rgbt_cc import (
    RGBTCCDset,
    RGBTCC_RGBDset,
    RGBTCC_TDset,
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    build_splits_rgbt_cc,
    seed_worker,
)
from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _set_requires_grad(module: nn.Module, req: bool) -> None:
    for p in module.parameters():
        p.requires_grad = req


@torch.no_grad()
def _count_from_density(den: torch.Tensor) -> torch.Tensor:
    # den: [B,1,H,W]
    return den.sum(dim = (2, 3))


@torch.no_grad()
def _partition_density(den: torch.Tensor, grid: int) -> torch.Tensor:
    """Return per-cell counts for GAMEk (grid x grid)."""
    b, _c, h, w = den.shape
    if grid <= 1:
        return den.sum(dim = (2, 3)).view(b, 1)

    gh = int(np.ceil(h / grid) * grid)
    gw = int(np.ceil(w / grid) * grid)
    if gh != h or gw != w:
        den = F.pad(den, (0, gw - w, 0, gh - h))

    _b, _c2, hp, wp = den.shape
    cell_h = hp // grid
    cell_w = wp // grid
    den = den.view(b, 1, grid, cell_h, grid, cell_w)
    den = den.sum(dim = (3, 5))
    return den.view(b, grid * grid)


@torch.inference_mode()
def eval_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    mae = 0.0
    rmse = 0.0
    game1 = 0.0
    game2 = 0.0
    game3 = 0.0
    n = 0

    for batch in loader:
        if len(batch) == 5:
            x_rgb, x_t, den_gt, _img_id, _meta = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t = x_t.to(device, non_blocking = True)
            den_gt = den_gt.to(device, non_blocking = True)

            if amp:
                with torch.autocast(device_type = "cuda", dtype = torch.float16):
                    den_pred = model(x_rgb, x_t)
            else:
                den_pred = model(x_rgb, x_t)
        else:
            x, den_gt, _img_id, _meta = batch
            x = x.to(device, non_blocking = True)
            den_gt = den_gt.to(device, non_blocking = True)

            if amp:
                with torch.autocast(device_type = "cuda", dtype = torch.float16):
                    den_pred = model(x)
            else:
                den_pred = model(x)

        # Metrics in fp32 for stability
        den_pred = den_pred.float()
        den_gt = den_gt.float()

        cnt_pred = _count_from_density(den_pred).cpu().numpy()
        cnt_gt = _count_from_density(den_gt).cpu().numpy()

        err = np.abs(cnt_pred - cnt_gt)
        mae += float(err.sum())
        rmse += float((err ** 2).sum())

        g1 = (_partition_density(den_pred, 2) - _partition_density(den_gt, 2)).abs().sum(dim = 1).cpu().numpy()
        g2 = (_partition_density(den_pred, 4) - _partition_density(den_gt, 4)).abs().sum(dim = 1).cpu().numpy()
        g3 = (_partition_density(den_pred, 8) - _partition_density(den_gt, 8)).abs().sum(dim = 1).cpu().numpy()

        game1 += float(g1.sum())
        game2 += float(g2.sum())
        game3 += float(g3.sum())
        n += int(cnt_gt.shape[0])

    if n == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "game1": float("nan"),
            "game2": float("nan"),
            "game3": float("nan"),
        }

    mae /= n
    rmse = (rmse / n) ** 0.5
    game1 /= n
    game2 /= n
    game3 /= n
    return {"mae": mae, "rmse": rmse, "game1": game1, "game2": game2, "game3": game3}


def _load_into(net: nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    if not ckpt_path:
        return
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location = map_location)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "network"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    sd = {kk.replace("module.", ""): vv for kk, vv in ckpt.items()}

    # If a full-model ckpt is passed, try to strip a known prefix.
    for prefix in ("rgb_net.", "t_net."):
        if any(k.startswith(prefix) for k in sd.keys()):
            sd2 = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if len(sd2) >= 10:
                sd = sd2
            break

    res = net.load_state_dict(sd, strict = False)
    print(f"[init] warm-start from {ckpt_path} (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev = False)
    ap.add_argument("--data_root", type = str, required = True)
    ap.add_argument("--out_dir", type = str, required = True)

    ap.add_argument("--mode", type = str, default = "late", choices = ["rgb", "t", "early", "late", "adaptive_late"])
    ap.add_argument("--epochs", type = int, default = 100)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--workers", type = int, default = 4)

    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--deterministic", action = "store_true")

    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--use_onecycle", action = "store_true")
    ap.add_argument("--max_lr", type = float, default = 1e-5)

    # Optional warm-start for *both* late and adaptive_late (keeps comparisons fair when enabled)
    ap.add_argument("--init_rgb_ckpt", type = str, default = "")
    ap.add_argument("--init_t_ckpt", type = str, default = "")

    # Adaptive-late gate LR (separate param group)
    ap.add_argument("--gate_lr", type = float, default = 1e-5)
    ap.add_argument("--max_gate_lr", type = float, default = 1e-5)

    ap.add_argument("--freeze_backbones_epochs", type = int, default = 0)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--grad_accum", type = int, default = 1)
    ap.add_argument("--clip_grad", type = float, default = 0.0)

    ap.add_argument("--crop_size", type = int, default = 224)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--down", type = int, default = 8)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    set_seed(args.seed, deterministic = bool(args.deterministic))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    base_train, base_val = build_splits_rgbt_cc(args.data_root)

    is_train_det = bool(args.deterministic)

    # Train datasets (224 crops by default)
    if args.mode == "rgb":
        ds_train = RGBTCC_RGBDset(base_train, crop_size = args.crop_size, sigma = args.sigma, down = args.down, is_train = True, deterministic = is_train_det)
    elif args.mode == "t":
        ds_train = RGBTCC_TDset(base_train, crop_size = args.crop_size, sigma = args.sigma, down = args.down, is_train = True, deterministic = is_train_det)
    else:
        ds_train = RGBTCCDset(base_train, crop_size = args.crop_size, sigma = args.sigma, down = args.down, is_train = True, deterministic = is_train_det)

    # Val is full-image, deterministic
    data_root_p = Path(args.data_root)
    val_split = "val" if (data_root_p / "val").exists() else "test"

    if args.mode == "rgb":
        ds_val = RGBTCC_RGBDataset(root = args.data_root, split = val_split, img_size = (768, 1024), sigma = args.sigma, return_pts = False, out_stride = args.down)
    elif args.mode == "t":
        ds_val = RGBTCC_TDataset(root = args.data_root, split = val_split, img_size = (768, 1024), sigma = args.sigma, return_pts = False, out_stride = args.down)
    else:
        ds_val = RGBTCC_PairedDataset(root = args.data_root, split = val_split, img_size = (768, 1024), sigma = args.sigma, return_pts = False, out_stride = args.down)

    g_train = torch.Generator()
    g_train.manual_seed(args.seed)
    g_val = torch.Generator()
    g_val.manual_seed(args.seed + 1)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g_train,
    )

    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g_val,
    )

    print(f"[init] train = {len(ds_train)}  val = {len(ds_val)}  workers = {args.workers}")

    # Model selection
    if args.mode == "rgb":
        model: nn.Module = CSRNet(load_imagenet = True)
    elif args.mode == "t":
        model = CSRNet(load_imagenet = True)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = True)
    elif args.mode == "late":
        model = CSRNetRGBT_Late(load_imagenet = True)
        _load_into(model.rgb_net, args.init_rgb_ckpt)
        _load_into(model.t_net, args.init_t_ckpt)
    elif args.mode == "adaptive_late":
        try:
            model = CSRNetRGBT_AdaptiveLate(load_imagenet = True)
        except TypeError:
            model = CSRNetRGBT_AdaptiveLate(load_weights = True)

        # Optional warm-start for adaptive experts.
        _load_into(model.rgb_net, args.init_rgb_ckpt)
        _load_into(model.t_net, args.init_t_ckpt)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    model = model.to(device)

    def loss_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Always compute loss in fp32 for stability (even under autocast).
        return F.mse_loss(pred.float(), gt.float(), reduction = "mean")

    try:
        scaler = GradScaler("cuda", enabled = bool(args.amp))
    except TypeError:
        scaler = GradScaler(enabled = bool(args.amp))

    # Optimizer and LR schedule
    if args.mode == "adaptive_late":
        backbone_params = list(model.rgb_net.parameters()) + list(model.t_net.parameters())
        gate_module = getattr(model, "gate", None) or getattr(model, "gate_net", None)
        if gate_module is None:
            raise AttributeError("CSRNetRGBT_AdaptiveLate must expose a gate module as .gate or .gate_net")
        gate_params = list(gate_module.parameters())
        params = [
            {"params": backbone_params, "lr": args.lr},
            {"params": gate_params, "lr": args.gate_lr},
        ]
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr = args.lr, weight_decay = args.weight_decay)

    if args.use_onecycle:
        max_lrs = [args.max_lr, args.max_gate_lr] if args.mode == "adaptive_late" else args.max_lr
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr = max_lrs,
            total_steps = args.epochs * max(1, len(dl_train)),
            pct_start = 0.1,
            div_factor = 10.0,
            final_div_factor = 100.0,
            anneal_strategy = "cos",
        )
    else:
        sch = None

    best_mae = float("inf")
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        if args.mode == "adaptive_late" and args.freeze_backbones_epochs > 0:
            _set_requires_grad(model.rgb_net, ep > args.freeze_backbones_epochs)
            _set_requires_grad(model.t_net, ep > args.freeze_backbones_epochs)
            gate_module = getattr(model, "gate", None) or getattr(model, "gate_net", None)
            if gate_module is None:
                raise AttributeError("CSRNetRGBT_AdaptiveLate must expose a gate module as .gate or .gate_net")
            _set_requires_grad(gate_module, True)

        running = 0.0
        step = 0
        opt.zero_grad(set_to_none = True)

        for batch in dl_train:
            step += 1

            if args.mode in ["early", "late", "adaptive_late"]:
                x_rgb, x_t, den_gt, _img_id, _meta = batch
                x_rgb = x_rgb.to(device, non_blocking = True)
                x_t = x_t.to(device, non_blocking = True)
                den_gt = den_gt.to(device, non_blocking = True)

                if args.amp:
                    with torch.autocast(device_type = "cuda", dtype = torch.float16):
                        den_pred = model(x_rgb, x_t)
                        loss = loss_fn(den_pred, den_gt)
                else:
                    den_pred = model(x_rgb, x_t)
                    loss = loss_fn(den_pred, den_gt)

            else:
                x, den_gt, _img_id, _meta = batch
                x = x.to(device, non_blocking = True)
                den_gt = den_gt.to(device, non_blocking = True)

                if args.amp:
                    with torch.autocast(device_type = "cuda", dtype = torch.float16):
                        den_pred = model(x)
                        loss = loss_fn(den_pred, den_gt)
                else:
                    den_pred = model(x)
                    loss = loss_fn(den_pred, den_gt)

            loss = loss / max(1, int(args.grad_accum))

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % max(1, int(args.grad_accum))) == 0:
                if args.clip_grad and args.clip_grad > 0:
                    if args.amp:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                if args.amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none = True)

                if sch is not None:
                    sch.step()

            running += float(loss.detach().item()) * max(1, int(args.grad_accum))

            if step == 1 or (step % 50) == 0:
                print(f"[e{ep:03d} s{step:04d}/{len(dl_train)}] loss = {running / step:.6f}")

        train_loss = running / max(1, step)

        metrics = eval_one_epoch(model, dl_val, device, amp = bool(args.amp))
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        g1 = metrics["game1"]
        g2 = metrics["game2"]
        g3 = metrics["game3"]

        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"Epoch {ep:03d}: train_loss = {train_loss:.6f}  "
            f"MAE = {mae:.3f}  RMSE = {rmse:.3f}  "
            f"GAME1 = {g1:.3f}  GAME2 = {g2:.3f}  GAME3 = {g3:.3f}  "
            f"lr = {lr_now:.2e}  time = {dt:.1f}s"
        )

        ckpt = {
            "epoch": ep,
            "args": vars(args),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_mae": best_mae,
        }
        torch.save(ckpt, last_path)

        if mae < best_mae:
            best_mae = mae
            torch.save(ckpt, best_path)

    print(f"[done] best_mae = {best_mae:.3f}  best_path = {best_path}")


if __name__ == "__main__":
    main()
