# scripts/eval_rgbt.py — Early-fusion RGBT evaluator with MAE/RMSE + GAME(0–3)

import os, argparse, csv, math
import numpy as np
import cv2
import torch

from datasets.rgbt_cc import RGBTCC_RGBTDataset, load_points
from models.csrnet_rgbt import CSRNetRGBT  # early-fusion CSRNet variant

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def load_state_flex(model, ckpt_path, map_location):
    state = torch.load(ckpt_path, map_location = map_location)
    if isinstance(state, dict):
        for k in ["model", "state_dict", "ema", "net", "model_state_dict"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    missing, unexpected = model.load_state_dict(state, strict = False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5) -> {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5) -> {unexpected[:5]}")

def _cell_bounds(n, size):
    edges = np.linspace(0, size, n + 1)
    edges = np.round(edges).astype(int)
    edges[-1] = size
    return edges

def game_per_image(den_up, pts_xy_eval, W, H, levels = (0, 1, 2, 3)):
    errs = []
    for L in levels:
        g = 2 ** L
        xs = _cell_bounds(g, W)
        ys = _cell_bounds(g, H)

        # GT counts per cell
        if pts_xy_eval.size == 0:
            gt_grid = np.zeros((g, g), dtype = np.float32)
        else:
            xx = np.clip(pts_xy_eval[:, 0], 0, W - 1e-3)
            yy = np.clip(pts_xy_eval[:, 1], 0, H - 1e-3)
            bx = np.searchsorted(xs, xx, side = "right") - 1
            by = np.searchsorted(ys, yy, side = "right") - 1
            gt_grid = np.zeros((g, g), dtype = np.float32)
            np.add.at(gt_grid, (by, bx), 1.0)

        # Pred counts per cell
        pred_grid = np.zeros((g, g), dtype = np.float32)
        for r in range(g):
            y0, y1 = ys[r], ys[r + 1]
            for c in range(g):
                x0, x1 = xs[c], xs[c + 1]
                pred_grid[r, c] = float(den_up[y0:y1, x0:x1].sum())

        errs.append(np.abs(pred_grid - gt_grid).sum())
    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required = True, help = "data/RGBT-CC-CVPR2021")
    ap.add_argument("--split", choices = ["val", "test"], default = "val")
    ap.add_argument("--ckpt", required = True, help = "path to .pth or to a folder containing checkpoints")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--save_csv", required = True)
    ap.add_argument("--save_vis", required = True)
    ap.add_argument("--no_vis", action = "store_true")
    ap.add_argument("--levels", type = str, default = "0,1,2,3")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok = True)
    os.makedirs(args.save_vis, exist_ok = True)

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip() != "")
    W_eval, H_eval = args.img_w, args.img_h

    device = get_device()
    model  = CSRNetRGBT(load_imagenet = False).to(device)

    ckpt_path = os.path.expanduser(args.ckpt)
    if os.path.isdir(ckpt_path):
        cands = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith((".pth", ".pt"))]
        if not cands:
            raise FileNotFoundError(f"No .pth/.pt found in directory: {ckpt_path}")
        ckpt_path = max(cands, key = os.path.getmtime)
        print(f"[info] Using newest checkpoint in dir: {ckpt_path}")
    elif not os.path.isfile(ckpt_path):
        alt = os.path.join(ckpt_path, "best.pth")
        if os.path.isfile(alt):
            ckpt_path = alt
            print(f"[info] Using {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    load_state_flex(model, ckpt_path, map_location = device)
    model.eval()

    ds = RGBTCC_RGBTDataset(args.data_root, args.split, (H_eval, W_eval), args.sigma)

    header = ["image_id", "gt_count", "pred_count", "abs_err"] + [f"GAME{L}" for L in levels]
    rows = [tuple(header)]

    N = len(ds)
    mae = 0.0
    rmse = 0.0
    game_acc = np.zeros(len(levels), dtype = np.float64)

    mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype = np.float32)

    with torch.no_grad():
        for sample in ds:
            # Accept either 5-tuple or 4-tuple from the dataset
            if len(sample) == 5:
                rgb_t, t_t, den_gt, name, gt = sample
            else:
                rgb_t, t_t, name, gt = sample
                den_gt = None

            x = torch.cat([rgb_t, t_t], dim = 0).unsqueeze(0).to(device)  # (1,4,H,W)
            den_pred = model(x).squeeze().cpu().numpy()

            # Clamp tiny negatives to zero for stability
            c_pred = max(0.0, float(den_pred.sum()))
            err = abs(c_pred - gt)
            mae += err
            rmse += (c_pred - gt) ** 2

            # Upsample predicted density to eval size for GAME
            den_up = cv2.resize(den_pred, (W_eval, H_eval), interpolation = cv2.INTER_CUBIC)

            # Load GT points and scale to eval size
            sid = os.path.splitext(name)[0]
            split_dir = ds.split_dir
            gt_no_ext = os.path.join(split_dir, f"{sid}_GT")
            pts = load_points(gt_no_ext)

            rgb_path = os.path.join(split_dir, f"{sid}_RGB.jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(split_dir, f"{sid}_RGB.png")
            orig = cv2.imread(rgb_path)
            if orig is None:
                H0, W0 = H_eval, W_eval  # fallback
            else:
                H0, W0 = orig.shape[:2]

            if pts.size > 0:
                pts = pts.copy()
                pts[:, 0] *= (W_eval / float(W0))
                pts[:, 1] *= (H_eval / float(H0))

            game_list = game_per_image(den_up, pts, W_eval, H_eval, levels)
            game_acc += np.array(game_list, dtype = np.float64)

            if not args.no_vis:
                m = float(den_up.max())
                if m <= 1e-6:
                    dm_vis = np.zeros_like(den_up, dtype = np.uint8)
                else:
                    dm_vis = (den_up / m * 255.0).astype(np.uint8)
                heat = cv2.applyColorMap(dm_vis, cv2.COLORMAP_JET)

                # Reconstruct RGB from normalized tensor
                img_np = rgb_t.permute(1, 2, 0).cpu().numpy()
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0.0, 1.0)
                base_bgr = (img_np[:, :, ::-1] * 255.0).astype(np.uint8)

                overlay = cv2.addWeighted(base_bgr, 0.6, heat, 0.4, 0.0)
                cv2.putText(overlay, f"GT: {gt:.0f}  Pred: {c_pred:.1f}",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(args.save_vis, f"vis_{sid}.png"), overlay)

            rows.append((name, f"{gt:.1f}", f"{c_pred:.2f}", f"{err:.2f}",
                         *[f"{g:.2f}" for g in game_list]))

    mae /= max(1, N)
    rmse = math.sqrt(rmse / max(1, N))
    game_avg = (game_acc / max(1, N)).tolist()

    print(f"[{args.split}]  MAE = {mae:.2f}  RMSE = {rmse:.2f}  "
          + "  ".join([f"GAME{L} = {g:.2f}" for L, g in zip(levels, game_avg)])
          + f"  N = {N}")

    rows.append(("AVERAGE", f"{mae:.2f}", f"{rmse:.2f}", f"{mae:.2f}",
                 *[f"{g:.2f}" for g in game_avg]))
    with open(args.save_csv, "w", newline = "") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved CSV to {args.save_csv} and overlays to {args.save_vis}")

if __name__ == "__main__":
    main()