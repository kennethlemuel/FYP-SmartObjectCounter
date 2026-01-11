import os
import sys
import time
import math
import argparse
import random
import re
import copy
from contextlib import nullcontext
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    _THIS_DIR,
    os.path.dirname(_THIS_DIR),
    os.path.dirname(os.path.dirname(_THIS_DIR)),
]
PROJECT_ROOT = None
for c in _CANDIDATES:
    if os.path.isdir(os.path.join(c, "models")) and os.path.isdir(os.path.join(c, "datasets")):
        PROJECT_ROOT = c
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.path.dirname(_THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.csrnet import CSRNet
from models.rgbt_late import CSRNetRGBT_Late
from models.rgbt_adaptive_late import CSRNetRGBT_AdaptiveLate

from datasets.rgbt_cc import (
    RGBTCC_RGBDataset,
    RGBTCC_TDataset,
    RGBTCC_PairedDataset,
    RGBTCC_EarlyFusionDataset,
    density_from_points,
)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type = "cuda", dtype = torch.float16, enabled = True)
    return nullcontext()


def resize_density_sum_preserving(den: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    old_h, old_w = den.shape[-2], den.shape[-1]
    new_h, new_w = int(size_hw[0]), int(size_hw[1])

    if old_h == new_h and old_w == new_w:
        return den

    den_rs = F.interpolate(den, size = (new_h, new_w), mode = "bilinear", align_corners = False)
    den_rs = den_rs * (old_h * old_w) / float(new_h * new_w)
    return den_rs


def game_error(pred: torch.Tensor, gt: torch.Tensor, level: int) -> float:
    assert pred.ndim == 4 and gt.ndim == 4
    b, _, h, w = pred.shape
    k = 2 ** int(level)
    gh = max(1, h // k)
    gw = max(1, w // k)

    h2 = gh * k
    w2 = gw * k
    pred_c = pred[:, :, :h2, :w2].contiguous()
    gt_c = gt[:, :, :h2, :w2].contiguous()

    pred_cells = pred_c.view(b, 1, k, gh, k, gw).sum(dim = (3, 5))
    gt_cells = gt_c.view(b, 1, k, gh, k, gw).sum(dim = (3, 5))

    err = torch.abs(pred_cells - gt_cells).sum(dim = (2, 3))
    return float(err.mean().item())


def grid_count_loss(pred_pos: torch.Tensor, gt: torch.Tensor, level: int) -> torch.Tensor:
    assert pred_pos.ndim == 4 and gt.ndim == 4
    b, _, h, w = pred_pos.shape
    k = 2 ** int(level)
    gh = max(1, h // k)
    gw = max(1, w // k)

    h2 = gh * k
    w2 = gw * k
    pred_c = pred_pos[:, :, :h2, :w2].contiguous()
    gt_c = gt[:, :, :h2, :w2].contiguous()

    pred_cells = pred_c.view(b, 1, k, gh, k, gw).sum(dim = (3, 5))
    gt_cells = gt_c.view(b, 1, k, gh, k, gw).sum(dim = (3, 5))

    err = torch.abs(pred_cells - gt_cells).sum(dim = (2, 3))
    return err.mean()


def parse_seq_frame(name: str) -> Tuple[str, int]:
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.match(r"^(.*?)(?:[_-])?(\d+)$", stem)
    if m is None:
        return stem, 0
    seq = m.group(1)
    seq = seq if len(seq) > 0 else stem
    frame = int(m.group(2))
    return seq, frame


class TemporalPairDataset(Dataset):
    def __init__(self, base: Dataset, pair_delta: int = 1):
        self.base = base
        self.pair_delta = max(1, int(pair_delta))

        if not hasattr(self.base, "ids"):
            raise ValueError("Base dataset must have .ids for temporal_pair without expensive preloading.")

        ids = list(self.base.ids)

        seq_to: Dict[str, List[Tuple[int, int]]] = {}
        for i, sid in enumerate(ids):
            seq, frame = parse_seq_frame(f"{sid}.jpg")
            seq_to.setdefault(seq, []).append((frame, i))

        self._pair_index: Dict[int, int] = {}
        for _seq, arr in seq_to.items():
            arr_sorted = sorted(arr, key = lambda x: x[0])
            idxs = [i for _, i in arr_sorted]
            n = len(idxs)
            for pos, idx in enumerate(idxs):
                j_pos = pos + self.pair_delta
                if j_pos >= n:
                    j_pos = pos - self.pair_delta
                if j_pos < 0 or j_pos >= n:
                    j_pos = pos
                self._pair_index[idx] = idxs[j_pos]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        j = self._pair_index.get(idx, idx)
        return self.base[idx], self.base[j]


def hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims = [-1])


def crop_params(h: int, w: int, crop: int, stride: int) -> Tuple[int, int, int, int]:
    ch = min(crop, h)
    cw = min(crop, w)

    ch = max(stride, (ch // stride) * stride)
    cw = max(stride, (cw // stride) * stride)

    h_out = h // stride
    w_out = w // stride
    ch_out = ch // stride
    cw_out = cw // stride

    oy0 = 0 if h_out <= ch_out else random.randint(0, h_out - ch_out)
    ox0 = 0 if w_out <= cw_out else random.randint(0, w_out - cw_out)

    y0 = oy0 * stride
    x0 = ox0 * stride
    y1 = y0 + ch
    x1 = x0 + cw
    return y0, y1, x0, x1


class TrainAugment(Dataset):
    def __init__(
        self,
        base: Dataset,
        mode: str,
        sigma: float,
        crop_size: int = 224,
        flip_prob: float = 0.5,
        stride: int = 8,
    ):
        self.base = base
        self.mode = mode
        self.crop_size = int(crop_size)
        self.flip_prob = float(flip_prob)
        self.stride = int(stride)
        self.sigma_out = float(sigma) / float(self.stride)

    def __len__(self) -> int:
        return len(self.base)

    def _crop_den_from_points(
        self,
        pts_out: torch.Tensor,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> torch.Tensor:
        oy0 = int(y0 // self.stride)
        oy1 = int(y1 // self.stride)
        ox0 = int(x0 // self.stride)
        ox1 = int(x1 // self.stride)

        h_out = max(1, oy1 - oy0)
        w_out = max(1, ox1 - ox0)

        if pts_out.numel() == 0:
            dm = np.zeros((h_out, w_out), dtype = np.float32)
            return torch.from_numpy(dm)[None, ...]

        pts = pts_out.clone()
        m = (pts[:, 0] >= ox0) & (pts[:, 0] < ox1) & (pts[:, 1] >= oy0) & (pts[:, 1] < oy1)
        pts_c = pts[m]
        if pts_c.numel() == 0:
            dm = np.zeros((h_out, w_out), dtype = np.float32)
            return torch.from_numpy(dm)[None, ...]

        pts_c[:, 0] -= float(ox0)
        pts_c[:, 1] -= float(oy0)

        dm = density_from_points(pts_c.cpu().numpy(), h_out, w_out, sigma = self.sigma_out)

        gt_n = float(pts_c.shape[0])
        s = float(dm.sum())
        if gt_n > 0.0 and s > 0.0:
            dm = dm * (gt_n / s)

        return torch.from_numpy(dm.astype(np.float32, copy = False))[None, ...]

    def _augment_one(self, sample):
        do_flip = (random.random() < self.flip_prob)

        if self.mode in ["rgb", "t"]:
            x, _den_full, pts_out, name, _gt_count = sample
            _, h, w = x.shape
            y0, y1, x0, x1 = crop_params(h, w, self.crop_size, self.stride)
            x = x[:, y0:y1, x0:x1]
            den = self._crop_den_from_points(pts_out, y0, y1, x0, x1)
            if do_flip:
                x = hflip(x)
                den = hflip(den)
            return x.contiguous(), den.contiguous(), name, float(den.sum().item())

        if self.mode == "early":
            x4, _den_full, pts_out, name, _gt_count = sample
            _, h, w = x4.shape
            y0, y1, x0, x1 = crop_params(h, w, self.crop_size, self.stride)
            x4 = x4[:, y0:y1, x0:x1]
            den = self._crop_den_from_points(pts_out, y0, y1, x0, x1)
            if do_flip:
                x4 = hflip(x4)
                den = hflip(den)
            return x4.contiguous(), den.contiguous(), name, float(den.sum().item())

        x_rgb, x_t3, _den_full, pts_out, name, _gt_count = sample
        _, h, w = x_rgb.shape
        y0, y1, x0, x1 = crop_params(h, w, self.crop_size, self.stride)
        x_rgb = x_rgb[:, y0:y1, x0:x1]
        x_t3 = x_t3[:, y0:y1, x0:x1]
        den = self._crop_den_from_points(pts_out, y0, y1, x0, x1)
        if do_flip:
            x_rgb = hflip(x_rgb)
            x_t3 = hflip(x_t3)
            den = hflip(den)
        return x_rgb.contiguous(), x_t3.contiguous(), den.contiguous(), name, float(den.sum().item())

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
            a = self._augment_one(item[0])
            b = self._augment_one(item[1])
            return a, b
        return self._augment_one(item)


def build_model(mode: str, load_imagenet: bool = True) -> nn.Module:
    if mode == "rgb":
        return CSRNet(load_imagenet = load_imagenet)
    if mode == "t":
        return CSRNet(load_imagenet = load_imagenet)
    if mode == "early":
        from models.rgbt_early import CSRNetRGBT_Early
        return CSRNetRGBT_Early(load_imagenet = load_imagenet)
    if mode == "late":
        return CSRNetRGBT_Late(load_imagenet = load_imagenet)
    if mode == "adaptive_late":
        return CSRNetRGBT_AdaptiveLate(load_imagenet = load_imagenet)
    raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mode: str, game_levels = (0, 1, 2, 3)):
    model.eval()
    rmse_acc = 0.0
    mae_acc = 0.0
    game_acc = {L: 0.0 for L in game_levels}

    n = 0
    for batch in loader:
        if mode == "rgb":
            x, den, _name, _gtc = batch
            x = x.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x)
        elif mode == "t":
            x, den, _name, _gtc = batch
            x = x.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x)
        elif mode == "early":
            x4, den, _name, _gtc = batch
            x4 = x4.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x4)
        else:
            x_rgb, x_t3, den, _name, _gtc = batch
            x_rgb = x_rgb.to(device, non_blocking = True)
            x_t3 = x_t3.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            pred = model(x_rgb, x_t3)

        pred = F.relu(pred)

        if pred.shape[-2:] != den.shape[-2:]:
            den = resize_density_sum_preserving(den, pred.shape[-2:])

        c_pred = float(pred.sum().item())
        c_gt = float(den.sum().item())
        err = (c_pred - c_gt)

        mae_acc += abs(err)
        rmse_acc += (err ** 2)

        for L in game_levels:
            game_acc[L] += game_error(pred, den, level = L)

        n += 1

    n = max(1, n)
    out = {"MAE": mae_acc / n, "RMSE": math.sqrt(rmse_acc / n)}
    for L in game_levels:
        out[f"GAME{L}"] = game_acc[L] / n
    return out


def make_optimizer(args, model: nn.Module):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def make_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "multistep":
        milestones = [int(x) for x in args.milestones.split(",") if len(x.strip()) > 0]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = args.gamma)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def illumination_gate_loss(aux: Dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    gate = aux["gate"].float()
    lum = aux["lum"].float()
    tau = float(tau)

    denom = max(1e-6, (1.0 - tau))
    target = ((lum - tau) / denom).clamp(0.0, 1.0)
    return F.mse_loss(gate, target)


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v_ema in esd.items():
        v = msd[k]
        if torch.is_floating_point(v_ema):
            v_ema.mul_(decay).add_(v, alpha = 1.0 - decay)
        else:
            v_ema.copy_(v)


def _count_loss(pred_pos: torch.Tensor, den: torch.Tensor, rel: bool, beta: float) -> torch.Tensor:
    pred_f = pred_pos.float()
    den_f = den.float()

    c_pred = pred_f.sum(dim = (1, 2, 3))
    c_gt = den_f.sum(dim = (1, 2, 3))

    if rel:
        denom = (c_gt.detach() + 1.0)
        c_pred = c_pred / denom
        c_gt = c_gt / denom

    return F.smooth_l1_loss(c_pred, c_gt, beta = float(beta))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer,
    scaler,
    args,
    epoch: int,
    ema_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    model.train()
    mse = nn.MSELoss(reduction = "mean")

    loss_acc = 0.0
    base_acc = 0.0
    temp_acc = 0.0
    illum_acc = 0.0
    count_acc = 0.0
    game_acc = 0.0

    step = 0
    optimizer.zero_grad(set_to_none = True)

    warm = max(0, int(args.aux_warmup_epochs))
    aux_ramp = 1.0 if warm == 0 else min(1.0, float(epoch) / float(warm))

    def forward_single(sample):
        if args.mode in ["rgb", "t"]:
            x, den, _name, _gtc = sample
            x = x.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            with autocast_ctx(device, enabled = bool(args.amp)):
                pred = model(x)  # IMPORTANT: no ReLU here
            return pred, den, None

        if args.mode == "early":
            x4, den, _name, _gtc = sample
            x4 = x4.to(device, non_blocking = True)
            den = den.to(device, non_blocking = True)
            with autocast_ctx(device, enabled = bool(args.amp)):
                pred = model(x4)  # IMPORTANT: no ReLU here
            return pred, den, None

        x_rgb, x_t3, den, _name, _gtc = sample
        x_rgb = x_rgb.to(device, non_blocking = True)
        x_t3 = x_t3.to(device, non_blocking = True)
        den = den.to(device, non_blocking = True)

        if args.mode == "adaptive_late":
            with autocast_ctx(device, enabled = bool(args.amp)):
                pred, aux = model.forward_with_aux(x_rgb, x_t3)  # IMPORTANT: no ReLU here
            return pred, den, aux

        with autocast_ctx(device, enabled = bool(args.amp)):
            pred = model(x_rgb, x_t3)  # IMPORTANT: no ReLU here
        return pred, den, None

    for batch in loader:
        step += 1

        is_pair = isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list))
        if is_pair:
            if args.batch_size != 1:
                raise ValueError("temporal_pair requires batch_size = 1.")

            a, b = batch
            pred_a, den_a, aux_a = forward_single(a)
            pred_b, den_b, aux_b = forward_single(b)

            if pred_a.shape[-2:] != den_a.shape[-2:]:
                den_a = resize_density_sum_preserving(den_a, pred_a.shape[-2:])
            if pred_b.shape[-2:] != den_b.shape[-2:]:
                den_b = resize_density_sum_preserving(den_b, pred_b.shape[-2:])

            pred_a_pos = F.relu(pred_a)
            pred_b_pos = F.relu(pred_b)

            with torch.autocast(device_type = "cuda", enabled = False):
                base_a = mse(pred_a.float(), den_a.float())
                base_b = mse(pred_b.float(), den_b.float())
                base_loss = 0.5 * (base_a + base_b)

                count_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_count > 0.0:
                    cl_a = _count_loss(pred_a_pos, den_a, rel = bool(args.count_loss_rel), beta = args.count_loss_beta)
                    cl_b = _count_loss(pred_b_pos, den_b, rel = bool(args.count_loss_rel), beta = args.count_loss_beta)
                    count_loss = 0.5 * (cl_a + cl_b)

                game_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_game > 0.0:
                    gl_a = grid_count_loss(pred_a_pos.float(), den_a.float(), level = int(args.game_level))
                    gl_b = grid_count_loss(pred_b_pos.float(), den_b.float(), level = int(args.game_level))
                    game_loss = 0.5 * (gl_a + gl_b)

                temp_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_temp > 0.0:
                    cpa = pred_a_pos.float().sum(dim = (1, 2, 3))
                    cpb = pred_b_pos.float().sum(dim = (1, 2, 3))
                    cga = den_a.float().sum(dim = (1, 2, 3))
                    cgb = den_b.float().sum(dim = (1, 2, 3))
                    delta_pred = (cpb - cpa)
                    delta_gt = (cgb - cga)

                    if args.normalize_temp_loss:
                        denom = (delta_gt.detach().abs() + 1.0)
                        delta_pred = delta_pred / denom
                        delta_gt = delta_gt / denom

                    temp_loss = F.l1_loss(delta_pred, delta_gt)

                illum_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_illum > 0.0 and args.mode == "adaptive_late" and aux_a is not None:
                    il_a = illumination_gate_loss(aux_a, tau = args.illum_tau)
                    il_b = il_a
                    if aux_b is not None:
                        il_b = illumination_gate_loss(aux_b, tau = args.illum_tau)
                    illum_loss = 0.5 * (il_a + il_b)

                loss_total = (
                    base_loss
                    + (args.lambda_count * count_loss)
                    + (args.lambda_game * game_loss)
                    + aux_ramp * ((args.lambda_temp * temp_loss) + (args.lambda_illum * illum_loss))
                )

        else:
            pred, den, aux = forward_single(batch)

            if pred.shape[-2:] != den.shape[-2:]:
                den = resize_density_sum_preserving(den, pred.shape[-2:])

            pred_pos = F.relu(pred)

            with torch.autocast(device_type = "cuda", enabled = False):
                base_loss = mse(pred.float(), den.float())

                count_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_count > 0.0:
                    count_loss = _count_loss(pred_pos, den, rel = bool(args.count_loss_rel), beta = args.count_loss_beta)

                game_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_game > 0.0:
                    game_loss = grid_count_loss(pred_pos.float(), den.float(), level = int(args.game_level))

                temp_loss = torch.zeros((), device = device, dtype = torch.float32)

                illum_loss = torch.zeros((), device = device, dtype = torch.float32)
                if args.lambda_illum > 0.0 and args.mode == "adaptive_late" and aux is not None:
                    illum_loss = illumination_gate_loss(aux, tau = args.illum_tau)

                loss_total = (
                    base_loss
                    + (args.lambda_count * count_loss)
                    + (args.lambda_game * game_loss)
                    + aux_ramp * (args.lambda_illum * illum_loss)
                )

        loss = loss_total / float(args.grad_accum)

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        did_step = False
        if step % args.grad_accum == 0:
            if args.clip_grad > 0.0:
                if args.amp:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)

            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none = True)
            did_step = True

        if did_step and ema_model is not None:
            ema_update(ema_model, model, decay = float(args.ema_decay))

        loss_acc += float(loss_total.item())
        base_acc += float(base_loss.item())
        temp_acc += float(temp_loss.item())
        illum_acc += float(illum_loss.item())
        count_acc += float(count_loss.item())
        game_acc += float(game_loss.item())

    if (step % args.grad_accum) != 0:
        if args.clip_grad > 0.0:
            if args.amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)

        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none = True)

        if ema_model is not None:
            ema_update(ema_model, model, decay = float(args.ema_decay))

    n = max(1, len(loader))
    return {
        "loss": loss_acc / n,
        "base": base_acc / n,
        "temp": temp_acc / n,
        "illum": illum_acc / n,
        "count": count_acc / n,
        "game": game_acc / n,
    }


def save_ckpt(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_rmse: float,
    ema_model: Optional[nn.Module] = None,
):
    obj = {
        "epoch": int(epoch),
        "best_rmse": float(best_rmse),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    if ema_model is not None:
        obj["ema_model"] = ema_model.state_dict()
    torch.save(obj, path)


def load_resume(path: str, model: nn.Module, optimizer, scheduler, ema_model: Optional[nn.Module]):
    ck = torch.load(path, map_location = "cpu")
    model.load_state_dict(ck["model"], strict = True)
    if "optimizer" in ck and optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler", None) is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if ema_model is not None and ("ema_model" in ck):
        ema_model.load_state_dict(ck["ema_model"], strict = True)
    start_epoch = int(ck.get("epoch", 0))
    best_rmse = float(ck.get("best_rmse", float("inf")))
    return start_epoch, best_rmse


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

    # SOTA-aligned defaults
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--optimizer", choices = ["adam", "adamw"], default = "adam")

    # Scheduler is not mandated by the paper, so default to none for cleaner comparability
    ap.add_argument("--scheduler", choices = ["none", "multistep"], default = "none")
    ap.add_argument("--milestones", type = str, default = "200,300")
    ap.add_argument("--gamma", type = float, default = 0.1)

    # SOTA-aligned augmentation defaults
    ap.add_argument("--crop_size", type = int, default = 224)
    ap.add_argument("--flip_prob", type = float, default = 0.5)

    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--clip_grad", type = float, default = 0.0)

    ap.add_argument("--num_workers", type = int, default = -1)
    ap.add_argument("--seed", type = int, default = 42)

    ap.add_argument("--save_dir", required = True)
    ap.add_argument("--resume", type = str, default = "")

    # Novelty controls (keep off for baselines unless doing ablations)
    ap.add_argument("--temporal_pair", action = "store_true")
    ap.add_argument("--pair_delta", type = int, default = 1)
    ap.add_argument("--lambda_temp", type = float, default = 0.0)
    ap.add_argument("--normalize_temp_loss", action = "store_true")
    ap.add_argument("--aux_warmup_epochs", type = int, default = 10)

    ap.add_argument("--lambda_illum", type = float, default = 0.0)
    ap.add_argument("--illum_tau", type = float, default = 0.35)

    ap.add_argument("--lambda_count", type = float, default = 0.0)
    ap.add_argument("--count_loss_rel", action = "store_true")
    ap.add_argument("--count_loss_beta", type = float, default = 1.0)

    ap.add_argument("--lambda_game", type = float, default = 0.0)
    ap.add_argument("--game_level", type = int, default = 1)

    ap.add_argument("--ema_decay", type = float, default = 0.0)  # 0 disables
    ap.add_argument("--ema_eval", action = "store_true")

    args = ap.parse_args()

    assert args.img_h % args.out_stride == 0, "img_h must be divisible by out_stride"
    assert args.img_w % args.out_stride == 0, "img_w must be divisible by out_stride"

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(args.seed, deterministic = True)

    device = get_device()
    print(f"[init] device = {device}")
    print(f"[init] PROJECT_ROOT = {PROJECT_ROOT}")

    img_size = (args.img_h, args.img_w)

    if args.mode == "rgb":
        base_train = RGBTCC_RGBDataset(args.data_root, args.split_train, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
        base_val = RGBTCC_RGBDataset(args.data_root, args.split_val, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)
    elif args.mode == "t":
        base_train = RGBTCC_TDataset(args.data_root, args.split_train, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
        base_val = RGBTCC_TDataset(args.data_root, args.split_val, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)
    elif args.mode == "early":
        base_train = RGBTCC_EarlyFusionDataset(args.data_root, args.split_train, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
        base_val = RGBTCC_EarlyFusionDataset(args.data_root, args.split_val, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)
    else:
        base_train = RGBTCC_PairedDataset(args.data_root, args.split_train, img_size, args.sigma, return_pts = True, out_stride = args.out_stride)
        base_val = RGBTCC_PairedDataset(args.data_root, args.split_val, img_size, args.sigma, return_pts = False, out_stride = args.out_stride)

    if args.temporal_pair:
        if args.mode != "adaptive_late":
            print("[warn] temporal_pair is mainly intended for adaptive_late. Using it for baselines is an ablation.")
        base_train = TemporalPairDataset(base_train, pair_delta = args.pair_delta)

    train_ds = TrainAugment(
        base_train,
        mode = args.mode,
        sigma = args.sigma,
        crop_size = args.crop_size,
        flip_prob = args.flip_prob,
        stride = args.out_stride,
    )

    if args.num_workers < 0:
        args.num_workers = min(8, os.cpu_count() or 4)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
        generator = g,
    )

    val_loader = DataLoader(
        base_val,
        batch_size = 1,
        shuffle = False,
        num_workers = max(1, args.num_workers // 2),
        pin_memory = True,
        drop_last = False,
    )

    model = build_model(args.mode, load_imagenet = True).to(device)

    ema_model = None
    if args.ema_decay and args.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled = bool(args.amp))
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled = bool(args.amp))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled = False)

    start_epoch = 0
    best_rmse = float("inf")
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_rmse = load_resume(args.resume, model, optimizer, scheduler, ema_model)
        print(f"[resume] from {args.resume} | start_epoch = {start_epoch} | best_rmse = {best_rmse}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, device, optimizer, scaler, args, epoch = epoch, ema_model = ema_model
        )

        if scheduler is not None:
            scheduler.step()

        eval_model = ema_model if (ema_model is not None and args.ema_eval) else model
        val_stats = evaluate(eval_model, val_loader, device, mode = args.mode)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"lr = {lr_now:.2e} | "
            f"train loss = {train_stats['loss']:.4f} "
            f"(base {train_stats['base']:.4f}, count {train_stats['count']:.4f}, game {train_stats['game']:.4f}, "
            f"temp {train_stats['temp']:.4f}, illum {train_stats['illum']:.4f}) | "
            f"val RMSE = {val_stats['RMSE']:.3f}, MAE = {val_stats['MAE']:.3f}, "
            f"GAME0 = {val_stats['GAME0']:.3f}, GAME1 = {val_stats['GAME1']:.3f}, "
            f"GAME2 = {val_stats['GAME2']:.3f}, GAME3 = {val_stats['GAME3']:.3f} | "
            f"time = {dt:.1f}s"
        )

        save_ckpt(os.path.join(args.save_dir, "ckpt_last.pt"), model, optimizer, scheduler, epoch, best_rmse, ema_model = ema_model)

        if val_stats["RMSE"] < best_rmse:
            best_rmse = float(val_stats["RMSE"])
            save_ckpt(os.path.join(args.save_dir, "ckpt_best.pt"), model, optimizer, scheduler, epoch, best_rmse, ema_model = ema_model)

    print(f"[done] best RMSE = {best_rmse:.3f}")


if __name__ == "__main__":
    main()
