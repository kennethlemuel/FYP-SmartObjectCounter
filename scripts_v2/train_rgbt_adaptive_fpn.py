import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from datasets.rgbt_cc import RGBTCCDset, RGBTCC_PairedDataset, build_splits_rgbt_cc
from models_v2.rgbt_adaptive_fpn import AdaptiveFPNRGBT


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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _set_requires_grad(module: nn.Module, req: bool) -> None:
    for p in module.parameters():
        p.requires_grad = req


@torch.no_grad()
def _partition_density(den: torch.Tensor, grid: int) -> torch.Tensor:
    b, c, h, w = den.shape
    if grid <= 1:
        return den.sum(dim = (2, 3)).view(b, 1)
    gh = int(np.ceil(h / grid) * grid)
    gw = int(np.ceil(w / grid) * grid)
    if gh != h or gw != w:
        den = F.pad(den, (0, gw - w, 0, gh - h))
    _, _, hp, wp = den.shape
    cell_h = hp // grid
    cell_w = wp // grid
    den = den.view(b, 1, grid, cell_h, grid, cell_w)
    den = den.sum(dim = (3, 5))
    den = den.view(b, grid * grid)
    return den


@torch.no_grad()
def eval_one_epoch(model, loader, device, amp: bool):
    model.eval()
    mae = 0.0
    rmse = 0.0
    game1 = 0.0
    game2 = 0.0
    game3 = 0.0
    n = 0

    for batch in loader:
        x_rgb, x_t, den_gt, _img_id, _meta = batch
        x_rgb = x_rgb.to(device, non_blocking = True)
        x_t = x_t.to(device, non_blocking = True)
        den_gt = den_gt.to(device, non_blocking = True)

        if amp:
            with torch.autocast(device_type = "cuda", dtype = torch.float16):
                den_pred = model(x_rgb, x_t)
        else:
            den_pred = model(x_rgb, x_t)

        den_pred = torch.nan_to_num(den_pred.float(), nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)
        den_gt = torch.nan_to_num(den_gt.float(), nan = 0.0, posinf = 0.0, neginf = 0.0).clamp_min(0.0)

        cnt_pred = den_pred.double().sum(dim = (2, 3)).cpu().numpy()
        cnt_gt = den_gt.double().sum(dim = (2, 3)).cpu().numpy()

        err = np.abs(cnt_pred - cnt_gt)
        mae += float(err.sum())
        rmse += float((err ** 2).sum())

        g1 = (_partition_density(den_pred, 2) - _partition_density(den_gt, 2)).abs().sum(dim = 1).cpu().numpy()
        g2 = (_partition_density(den_pred, 4) - _partition_density(den_gt, 4)).abs().sum(dim = 1).cpu().numpy()
        g3 = (_partition_density(den_pred, 8) - _partition_density(den_gt, 8)).abs().sum(dim = 1).cpu().numpy()

        game1 += float(g1.sum())
        game2 += float(g2.sum())
        game3 += float(g3.sum())
        n += cnt_gt.shape[0]

    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "game1": float("nan"), "game2": float("nan"), "game3": float("nan")}

    mae /= n
    rmse = (rmse / n) ** 0.5
    game1 /= n
    game2 /= n
    game3 /= n
    return {"mae": mae, "rmse": rmse, "game1": game1, "game2": game2, "game3": game3}


def _strip_prefix(sd, prefix: str, subset_only: bool):
    if not prefix:
        return None
    matched = [k for k in sd.keys() if k.startswith(prefix)]
    if not matched:
        return None
    if subset_only:
        return {k[len(prefix):]: sd[k] for k in matched}
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _best_matching_state_dict(sd, target_keys, preferred_prefix: str):
    prefixes = []
    for prefix in (
        preferred_prefix,
        f"module.{preferred_prefix}" if preferred_prefix else "",
        f"model.{preferred_prefix}" if preferred_prefix else "",
        f"net.{preferred_prefix}" if preferred_prefix else "",
        "module.",
        "model.",
        "net.",
        "backbone.",
        "encoder.",
        "rgb.",
        "t.",
        "rgb_net.",
        "t_net.",
    ):
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)

    queue = [sd]
    seen = set()
    candidates = []
    while queue:
        cand = queue.pop(0)
        sig = tuple(sorted(cand.keys()))
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(cand)
        for prefix in prefixes:
            for subset_only in (True, False):
                trimmed = _strip_prefix(cand, prefix, subset_only = subset_only)
                if trimmed:
                    queue.append(trimmed)

    def score(cand):
        keys = set(cand.keys())
        overlap = len(keys & target_keys)
        extra = len(keys - target_keys)
        return (overlap, -extra, -len(keys))

    best = max(candidates, key = score)
    return best, score(best)


def load_backbone(net: nn.Module, ckpt_path: str, prefix: str):
    if not ckpt_path:
        return
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location = "cpu")
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
    sd = {kk[len("module."):] if kk.startswith("module.") else kk: vv for kk, vv in ckpt.items()}
    target_keys = set(net.state_dict().keys())
    sd, score = _best_matching_state_dict(sd, target_keys, prefix)
    overlap = score[0]
    res = net.load_state_dict(sd, strict = False)
    print(
        f"[init] warm-start {prefix} from {ckpt_path} "
        f"(overlap={overlap}/{len(target_keys)} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})"
    )


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha = 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type = str, required = True)
    ap.add_argument("--out_dir", type = str, required = True)

    ap.add_argument("--epochs", type = int, default = 400)
    ap.add_argument("--batch_size", type = int, default = 1)
    ap.add_argument("--workers", type = int, default = 4)

    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--deterministic", action = "store_true")

    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--use_onecycle", action = "store_true")
    ap.add_argument("--max_lr", type = float, default = 1e-4)

    ap.add_argument("--init_rgb_ckpt", type = str, default = "")
    ap.add_argument("--init_t_ckpt", type = str, default = "")
    ap.add_argument("--freeze_backbones_epochs", type = int, default = 0)

    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--grad_accum", type = int, default = 1)
    ap.add_argument("--clip_grad", type = float, default = 0.0)
    ap.add_argument("--lambda_cnt", type = float, default = 1e-3)

    ap.add_argument("--crop_size", type = int, default = 224)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--down", type = int, default = 4)

    ap.add_argument("--val_fullres", action = "store_true")
    ap.add_argument("--val_img_h", type = int, default = 768)
    ap.add_argument("--val_img_w", type = int, default = 1024)
    ap.add_argument("--modality_dropout", action = "store_true")
    ap.add_argument("--mdrop_prob", type = float, default = 0.1)
    ap.add_argument("--ema", action = "store_true")
    ap.add_argument("--ema_decay", type = float, default = 0.999)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    set_seed(args.seed, deterministic = bool(args.deterministic))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    base_train, base_val = build_splits_rgbt_cc(args.data_root)

    ds_train = RGBTCCDset(
        base_train,
        crop_size = args.crop_size,
        sigma = args.sigma,
        down = args.down,
        is_train = True,
        deterministic = bool(args.deterministic),
    )

    if args.val_fullres:
        val_split = "val"
        if not (Path(args.data_root) / "val").exists():
            val_split = "test"
        ds_val = RGBTCC_PairedDataset(
            root = args.data_root,
            split = val_split,
            img_size = (args.val_img_h, args.val_img_w),
            out_stride = args.down,
            sigma = args.sigma,
            return_pts = False,
        )
        print(f"[init] val_fullres = True (split={val_split}, img={args.val_img_h}x{args.val_img_w})")
    else:
        ds_val = RGBTCCDset(
            base_val,
            crop_size = args.crop_size,
            sigma = args.sigma,
            down = args.down,
            is_train = False,
            deterministic = True,
        )

    g = torch.Generator()
    g.manual_seed(args.seed)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g,
    )

    print(f"[init] train = {len(ds_train)}  val = {len(ds_val)}  workers = {args.workers}")

    model = AdaptiveFPNRGBT(load_imagenet = True).to(device)
    ema = EMA(model, decay = args.ema_decay) if args.ema else None

    if args.init_rgb_ckpt:
        load_backbone(model.rgb, args.init_rgb_ckpt, "rgb.")
    if args.init_t_ckpt:
        load_backbone(model.t, args.init_t_ckpt, "t.")

    def loss_fn(pred: torch.Tensor, gt: torch.Tensor, lambda_cnt: float = 1e-3) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan = 0.0, posinf = 0.0, neginf = 0.0)
        gt = torch.nan_to_num(gt, nan = 0.0, posinf = 0.0, neginf = 0.0)
        den_loss = F.mse_loss(pred, gt, reduction = "mean")
        if lambda_cnt > 0.0:
            pred_cnt = pred.sum(dim = (-2, -1))
            gt_cnt = gt.sum(dim = (-2, -1))
            cnt_loss = F.l1_loss(pred_cnt, gt_cnt, reduction = "mean")
            return den_loss + lambda_cnt * cnt_loss
        return den_loss

    scaler = GradScaler(enabled = bool(args.amp))

    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    if args.use_onecycle:
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr = args.max_lr,
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

        if args.freeze_backbones_epochs > 0:
            _set_requires_grad(model.rgb, ep > args.freeze_backbones_epochs)
            _set_requires_grad(model.t, ep > args.freeze_backbones_epochs)

        running = 0.0
        step = 0
        opt.zero_grad(set_to_none = True)

        for batch in dl_train:
            step += 1
            x_rgb, x_t, den_gt, _img_id, _meta = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t = x_t.to(device, non_blocking = True)
            den_gt = den_gt.to(device, non_blocking = True)

            if args.modality_dropout and args.mdrop_prob > 0:
                r = float(torch.rand(1).item())
                if r < args.mdrop_prob:
                    x_rgb = torch.zeros_like(x_rgb)
                elif r < 2.0 * args.mdrop_prob:
                    x_t = torch.zeros_like(x_t)

            if args.amp:
                with torch.autocast(device_type = "cuda", dtype = torch.float16):
                    den_pred = model(x_rgb, x_t)
                    loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)
            else:
                den_pred = model(x_rgb, x_t)
                loss = loss_fn(den_pred, den_gt, lambda_cnt = args.lambda_cnt)

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

                if ema is not None:
                    ema.update(model)

                opt.zero_grad(set_to_none = True)
                if sch is not None:
                    sch.step()

            running += float(loss.detach().item()) * max(1, int(args.grad_accum))

            if step == 1 or (step % 50) == 0:
                print(f"[e{ep:03d} s{step:04d}/{len(dl_train)}] loss = {running / step:.6f}")

        train_loss = running / max(1, step)
        if ema is not None:
            ema.apply_to(model)
            metrics = eval_one_epoch(model, dl_val, device, amp = bool(args.amp))
            ema.restore(model)
        else:
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
            f"MAE/GAME0 = {mae:.3f}  RMSE = {rmse:.3f}  "
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
            if ema is not None:
                ema.apply_to(model)
                ckpt["model"] = model.state_dict()
                ema.restore(model)
            torch.save(ckpt, best_path)

    print(f"[done] best_mae = {best_mae:.3f}  best_path = {best_path}")


if __name__ == "__main__":
    main()
