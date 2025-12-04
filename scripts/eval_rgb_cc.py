# scripts/eval_rgb_cc.py  — with GAME(0–3)
import os, argparse, csv, math
import numpy as np
import cv2
import torch
from models.csrnet import CSRNet
from datasets.rgbt_cc import RGBTCC_RGBDataset, load_points  # <- we reuse loader utils

def _cell_bounds(n, size):
    # n bins spanning [0, size]; returns integer edges len=n+1
    edges = np.linspace(0, size, n + 1)
    edges = np.round(edges).astype(int)
    edges[-1] = size
    return edges

def game_per_image(den_up, pts_xy_eval, W, H, levels=(0,1,2,3)):
    """
    GAME(L) = sum over grid cells of |pred_count(cell) - gt_count(cell)|
    (we return per-level image errors; the dataset average is mean of these)
    """
    errs = []
    # precompute GT histogram per level via vectorized binning
    for L in levels:
        g = 2 ** L
        xs = _cell_bounds(g, W)
        ys = _cell_bounds(g, H)

        # GT counts per cell
        if pts_xy_eval.size == 0:
            gt_grid = np.zeros((g, g), dtype=np.float32)
        else:
            # clamp and bin
            xx = np.clip(pts_xy_eval[:, 0], 0, W - 1e-3)
            yy = np.clip(pts_xy_eval[:, 1], 0, H - 1e-3)
            bx = np.searchsorted(xs, xx, side="right") - 1
            by = np.searchsorted(ys, yy, side="right") - 1
            gt_grid = np.zeros((g, g), dtype=np.float32)
            np.add.at(gt_grid, (by, bx), 1.0)

        # Pred counts per cell (sum of density map in each cell)
        pred_grid = np.zeros((g, g), dtype=np.float32)
        for r in range(g):
            y0, y1 = ys[r], ys[r + 1]
            for c in range(g):
                x0, x1 = xs[c], xs[c + 1]
                pred_grid[r, c] = float(den_up[y0:y1, x0:x1].sum())

        errs.append(np.abs(pred_grid - gt_grid).sum())
    return errs  # list length = len(levels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="data/RGBT-CC-CVPR2021")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_h", type=int, default=768)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--save_csv", required=True)
    ap.add_argument("--save_vis", required=True)
    ap.add_argument("--levels", type=str, default="0,1,2,3", help="GAME levels, e.g. 0,1,2,3")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    os.makedirs(args.save_vis, exist_ok=True)

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip() != "")
    W_eval, H_eval = args.img_w, args.img_h

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(load_imagenet=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = RGBTCC_RGBDataset(args.data_root, args.split, (H_eval, W_eval), args.sigma)

    # CSV header
    header = ["image_id", "gt_count", "pred_count", "abs_err"] + [f"GAME{L}" for L in levels]
    rows = [tuple(header)]

    N = len(ds)
    mae = rmse = 0.0
    game_acc = np.zeros(len(levels), dtype=np.float64)

    with torch.no_grad():
        for img_t, _, name, gt in ds:
            den_pred = model(img_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            c_pred = float(den_pred.sum())
            err = abs(c_pred - gt)
            mae += err
            rmse += (c_pred - gt) ** 2

            #upsample pred DM to eval size for GAME
            den_up = cv2.resize(den_pred, (W_eval, H_eval), interpolation=cv2.INTER_CUBIC)

            #recover the sample id and load GT points to compute GAME on same eval size
            sid = os.path.splitext(name)[0]
            split_dir = ds.split_dir
            gt_no_ext = os.path.join(split_dir, f"{sid}_GT")
            pts = load_points(gt_no_ext)
            #scale points from original image to eval size
            rgb_path = os.path.join(split_dir, f"{sid}_RGB.jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(split_dir, f"{sid}_RGB.png")
            orig = cv2.imread(rgb_path)
            H0, W0 = orig.shape[:2]
            if pts.size > 0:
                pts = pts.copy()
                pts[:, 0] *= (W_eval / float(W0))
                pts[:, 1] *= (H_eval / float(H0))

            #the GAME per image
            game_list = game_per_image(den_up, pts, W_eval, H_eval, levels)
            game_acc += np.array(game_list, dtype=np.float64)

            m = den_up.max()
            dm_vis = np.zeros_like(den_up, dtype=np.uint8) if m <= 1e-6 else (den_up / m * 255).astype(np.uint8)
            heat = cv2.applyColorMap(dm_vis, cv2.COLORMAP_JET)
            #reconstruct RGB from tensor for overlay
            img_np = (img_t.permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np, 0, 1)
            base_bgr = (img_np[:, :, ::-1] * 255.0).astype(np.uint8)
            overlay = cv2.addWeighted(base_bgr, 0.6, heat, 0.4, 0.0)
            cv2.putText(overlay, f"GT: {gt:.0f}  Pred: {c_pred:.1f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.save_vis, f"vis_{sid}.png"), overlay)

            rows.append((name, f"{gt:.1f}", f"{c_pred:.2f}", f"{err:.2f}", *[f"{g:.2f}" for g in game_list]))

    mae /= N
    rmse = math.sqrt(rmse / N)
    game_avg = (game_acc / N).tolist()

    print(f"[{args.split}]  MAE={mae:.2f}  RMSE={rmse:.2f}  " + "  ".join([f"GAME{L}={g:.2f}" for L, g in zip(levels, game_avg)]) + f"  N={N}")

    with open(args.save_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved CSV to {args.save_csv} and overlays to {args.save_vis}")
    print("Averages -> " + ", ".join([f"GAME{L}:{g:.2f}" for L, g in zip(levels, game_avg)]))

if __name__ == "__main__":
    main()