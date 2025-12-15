import argparse
import math
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.csrnet import CSRNet
from datasets.rgbt_cc import RGBTCC_TDataset

def to_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(0) == 1:
        return x.repeat(3, 1, 1)
    if x.dim() == 3 and x.size(0) == 3:
        return x
    raise ValueError(f"Unexpected thermal tensor shape: {tuple(x.shape)}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type = str, required = True)
    parser.add_argument("--ckpt", type = str, required = True)
    parser.add_argument("--split", type = str, default = "test", choices = ["train", "val", "test"])
    parser.add_argument("--img_h", type = int, default = 768)
    parser.add_argument("--img_w", type = int, default = 1024)
    parser.add_argument("--sigma", type = float, default = 15.0)
    parser.add_argument("--num_workers", type = int, default = 2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RGBTCC(
        data_root = args.data_root,
        split = args.split,
        img_h = args.img_h,
        img_w = args.img_w,
        sigma = args.sigma,
    )
    loader = DataLoader(ds, batch_size = 1, shuffle = False, num_workers = args.num_workers, pin_memory = True)

    model = CSRNet().to(device)
    ckpt = torch.load(args.ckpt, map_location = "cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict = True)
    model.eval()

    abs_err = []
    sq_err = []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            rgb, t, gt = batch[0], batch[1], batch[-1]
        elif isinstance(batch, dict):
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
        pred_cnt = pred.sum().item()
        gt_cnt = gt.sum().item()

        e = abs(pred_cnt - gt_cnt)
        abs_err.append(e)
        sq_err.append(e * e)

    mae = float(np.mean(abs_err)) if abs_err else 0.0
    rmse = float(math.sqrt(np.mean(sq_err))) if sq_err else 0.0

    print(f"[T-only] split={args.split} MAE={mae:.4f} RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()