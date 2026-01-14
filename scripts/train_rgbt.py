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
    density_from_points,
)


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


def _crop_params(h, w, crop, stride):
    ch = min(crop, h)
    cw = min(crop, w)

    ch = max(stride, (ch // stride) * stride)
    cw = max(stride, (cw // stride) * stride)

    max_y0 = h - ch
    max_x0 = w - cw

    y0 = 0 if max_y0 <= 0 else random.randint(0, max_y0 // stride) * stride
    x0 = 0 if max_x0 <= 0 else random.randint(0, max_x0 // stride) * stride

    return y0, y0 + ch, x0, x0 + cw


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def _resize_density_sum_preserving(den, size_hw):
    """
    Resize a density map to (H, W) while preserving the total sum (i.e., count).
    den: (B, 1, H, W) or (1, H, W) or (H, W)
    """
    if not torch.is_tensor(den):
        return den

    if den.dim() == 2:
        den4 = den[None, None, ...]
        squeeze_back = "2d"
    elif den.dim() == 3:
        den4 = den[None, ...]
        squeeze_back = "3d"
    else:
        den4 = den
        squeeze_back = None

    old_h, old_w = den4.shape[-2], den4.shape[-1]
    new_h, new_w = int(size_hw[0]), int(size_hw[1])

    if old_h == new_h and old_w == new_w:
        den_rs = den4
    else:
        den_rs = F.interpolate(den4, size = (new_h, new_w), mode = "bilinear", align_corners = False)
        den_rs = den_rs * (old_h * old_w) / float(new_h * new_w)

    if squeeze_back == "2d":
        return den_rs[0, 0]
    if squeeze_back == "3d":
        return den_rs[0]
    return den_rs


def _game(pred, gt, level = 0):
    b, _, h, w = pred.shape
    g = 2 ** int(level)

    h2 = (h // g) * g
    w2 = (w // g) * g
    if h2 == 0 or w2 == 0:
        c_pred = pred.sum(dim = (-2, -1)).view(-1)
        c_gt = gt.sum(dim = (-2, -1)).view(-1)
        return (c_pred - c_gt).abs().mean().item()

    pred = pred[:, :, :h2, :w2]
    gt = gt[:, :, :h2, :w2]

    hc = h2 // g
    wc = w2 // g

    pred_cells = pred.view(b, 1, g, hc, g, wc).sum(dim = (3, 5))
    gt_cells = gt.view(b, 1, g, hc, g, wc).sum(dim = (3, 5))

    err = (pred_cells - gt_cells).abs().view(b, -1).mean(dim = 1)
    return err.mean().item()


@torch.no_grad()
def evaluate(model, loader, device, mode, game_levels = (0, 1, 2, 3)):
    model.eval()
    rmse_acc = 0.0
    mae_acc = 0.0
    game_acc = {L: 0.0 for L in game_levels}

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
            den = _resize_density_sum_preserving(den, pred.shape[-2:])

        c_pred = float(pred.sum().item())
        c_gt = float(den.sum().item())
        err = (c_pred - c_gt)

        mae_acc += abs(err)
        rmse_acc += err ** 2

        for L in game_levels:
            game_acc[L] += _game(pred, den, level = L)

    n = max(1, len(loader))
    mae = mae_acc / n
    rmse = math.sqrt(rmse_acc / n)
    games = {L: game_acc[L] / n for L in game_levels}
    return mae, rmse, games


class MakeResizable(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        out = []
        for v in sample:
            if torch.is_tensor(v):
                out.append(_make_resizable(v))
            else:
                out.append(v)
        return tuple(out)


class TrainAugment(Dataset):
    """
    IMPORTANT: expects base_train to return points (pts_out) so we can
    regenerate density for each crop (avoids Gaussian truncation bug).
    """
    def __init__(self, base, mode, sigma, crop_size = 224, flip_prob = 0.5, stride = 8):
        self.base = base
        self.mode = mode
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.stride = int(stride)
        self.sigma_out = max(1.0, float(sigma) / float(self.stride))

    def __len__(self):
        return len(self.base)

    def _crop_and_make_den(self, pts_out, y0, y1, x0, x1):
        y0d, y1d = y0 // self.stride, y1 // self.stride
        x0d, x1d = x0 // self.stride, x1 // self.stride
        h_out = max(1, (y1 - y0) // self.stride)
        w_out = max(1, (x1 - x0) // self.stride)

        pts = pts_out
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        pts = np.asarray(pts, dtype = np.float32).reshape(-1, 2)

        if pts.size == 0:
            dm = np.zeros((h_out, w_out), dtype = np.float32)
        else:
            m = (
                (pts[:, 0] >= x0d) & (pts[:, 0] < x1d) &
                (pts[:, 1] >= y0d) & (pts[:, 1] < y1d)
            )
            pts_c = pts[m].copy()
            if pts_c.size == 0:
                dm = np.zeros((h_out, w_out), dtype = np.float32)
            else:
                pts_c[:, 0] -= x0d
                pts_c[:, 1] -= y0d
                dm = density_from_points(pts_c, h_out, w_out, sigma = self.sigma_out)

        den = torch.from_numpy(dm)[None, ...]
        return den

    def __getitem__(self, idx):
        sample = self.base[idx]
        do_flip = (random.random() < self.flip_prob)

        if self.mode == "rgb":
            x_rgb, den_full, pts_out, a, b = sample
            _, h, w = x_rgb.shape
            y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

            x_rgb = x_rgb[:, y0:y1, x0:x1]
            den = self._crop_and_make_den(pts_out, y0, y1, x0, x1)

            if do_flip:
                x_rgb = _hflip(x_rgb)
                den = _hflip(den)

            return _make_resizable(x_rgb), _make_resizable(den), a, b

        if self.mode == "t":
            x_t, den_full, pts_out, a, b = sample
            _, h, w = x_t.shape
            y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

            x_t = x_t[:, y0:y1, x0:x1]
            den = self._crop_and_make_den(pts_out, y0, y1, x0, x1)

            if do_flip:
                x_t = _hflip(x_t)
                den = _hflip(den)

            return _make_resizable(x_t), _make_resizable(den), a, b

        if self.mode == "early":
            x4, den_full, pts_out, a, b = sample
            _, h, w = x4.shape
            y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

            x4 = x4[:, y0:y1, x0:x1]
            den = self._crop_and_make_den(pts_out, y0, y1, x0, x1)

            if do_flip:
                x4 = _hflip(x4)
                den = _hflip(den)

            return _make_resizable(x4), _make_resizable(den), a, b

        x_rgb, x_t3, den_full, pts_out, a, b = sample
        _, h, w = x_rgb.shape
        y0, y1, x0, x1 = _crop_params(h, w, self.crop_size, stride = self.stride)

        x_rgb = x_rgb[:, y0:y1, x0:x1]
        x_t3 = x_t3[:, y0:y1, x0:x1]
        den = self._crop_and_make_den(pts_out, y0, y1, x0, x1)

        if do_flip:
            x_rgb = _hflip(x_rgb)
            x_t3 = _hflip(x_t3)
            den = _hflip(den)

        return _make_resizable(x_rgb), _make_resizable(x_t3), _make_resizable(den), a, b


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices = ["rgb", "t", "early", "late", "adaptive_late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split_train", default = "train")
    ap.add_argument("--split_val", default = "val")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)

    ap.add_argument("--out_stride", type = int, default = 8)

    ap.add_argument("--epochs", type = int, default = 400)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--grad_accum", type = int, default = 1)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--gate_lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)

    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--num_workers", type = int, default = -1)
    ap.add_argument("--seed", type = int, default = 42)

    ap.add_argument("--crop_size", type = int, default = 224)
    ap.add_argument("--flip_prob", type = float, default = 0.5)

    ap.add_argument("--clip_grad", type = float, default = 0.0)

    ap.add_argument("--optimizer", choices = ["adam", "adamw"], default = "adam")

    # Scheduler options
    ap.add_argument("--scheduler", choices = ["none", "multistep", "onecycle"], default = "none")
    ap.add_argument("--milestones", type = str, default = "200,300")
    ap.add_argument("--gamma", type = float, default = 0.1)

    # OneCycleLR options (fast change)
    ap.add_argument("--max_lr", type = float, default = 1e-4)
    ap.add_argument("--max_gate_lr", type = float, default = 1e-4)
    ap.add_argument("--pct_start", type = float, default = 0.1)
    ap.add_argument("--div_factor", type = float, default = 10.0)
    ap.add_argument("--final_div_factor", type = float, default = 1000.0)

    ap.add_argument("--freeze_backbones_epochs", type = int, default = 0)

    args = ap.parse_args()

    assert args.img_h % args.out_stride == 0, "img_h must be divisible by out_stride"
    assert args.img_w % args.out_stride == 0, "img_w must be divisible by out_stride"

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(args.seed, deterministic = True)

    device = get_device()
    print(f"[init] device = {device}")

    img_size = (args.img_h, args.img_w)

    if args.mode == "rgb":
        base_train = RGBTCC_RGBDataset(
            args.data_root, args.split_train, img_size, args.sigma,
            return_pts = True, out_stride = args.out_stride
        )
        base_val = RGBTCC_RGBDataset(
            args.data_root, args.split_val, img_size, args.sigma,
            return_pts = False, out_stride = args.out_stride
        )
    elif args.mode == "t":
        base_train = RGBTCC_TDataset(
            args.data_root, args.split_train, img_size, args.sigma,
            return_pts = True, out_stride = args.out_stride
        )
        base_val = RGBTCC_TDataset(
            args.data_root, args.split_val, img_size, args.sigma,
            return_pts = False, out_stride = args.out_stride
        )
    elif args.mode == "early":
        base_train = RGBTCC_EarlyFusionDataset(
            args.data_root, args.split_train, img_size, args.sigma,
            return_pts = True, out_stride = args.out_stride
        )
        base_val = RGBTCC_EarlyFusionDataset(
            args.data_root, args.split_val, img_size, args.sigma,
            return_pts = False, out_stride = args.out_stride
        )
    else:
        base_train = RGBTCC_PairedDataset(
            args.data_root, args.split_train, img_size, args.sigma,
            return_pts = True, out_stride = args.out_stride
        )
        base_val = RGBTCC_PairedDataset(
            args.data_root, args.split_val, img_size, args.sigma,
            return_pts = False, out_stride = args.out_stride
        )

    train_ds = TrainAugment(
        base_train,
        mode = args.mode,
        sigma = args.sigma,
        crop_size = args.crop_size,
        flip_prob = args.flip_prob,
        stride = args.out_stride,
    )
    val_ds = MakeResizable(base_val)

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

        # Handle remainder accumulation (if len(train_dl) not divisible by grad_accum)
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

        # Epoch-level scheduler step (MultiStepLR)
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