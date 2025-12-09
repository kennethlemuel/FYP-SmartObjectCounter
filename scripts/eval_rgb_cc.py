# scripts/eval_rgbt_cc.py
import os, argparse, csv, math
import numpy as np
import cv2
import torch
from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late  import CSRNetRGBT_Late
from datasets.rgbt_cc  import RGBTCC_TDataset, RGBTCC_PairedDataset, load_points

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
                state = state[k]; break
    model.load_state_dict(state, strict = False)

def _cell_bounds(n, size):
    edges = np.linspace(0, size, n + 1)
    edges = np.round(edges).astype(int)
    edges[-1] = size
    return edges

def game_per_image(den_up, pts_xy_eval, W, H, levels = (0, 1, 2, 3)):
    errs = []
    for L in levels:
        g = 2 ** L
        xs = _cell_bounds(g, W); ys = _cell_bounds(g, H)

        if pts_xy_eval.size == 0:
            gt_grid = np.zeros((g, g), dtype = np.float32)
        else:
            xx = np.clip(pts_xy_eval[:, 0], 0, W - 1e-3)
            yy = np.clip(pts_xy_eval[:, 1], 0, H - 1e-3)
            bx = np.searchsorted(xs, xx, side = "right") - 1
            by = np.searchsorted(ys, yy, side = "right") - 1
            gt_grid = np.zeros((g, g), dtype = np.float32)
            np.add.at(gt_grid, (by, bx), 1.0)

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
    ap.add_argument("--mode", choices = ["t", "early", "late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split", choices = ["val", "test"], default = "val")
    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--img_h", type = int, default = 640)
    ap.add_argument("--img_w", type = int, default = 960)
    ap.add_argument("--save_csv", required = True)
    ap.add_argument("--save_vis", required = True)
    ap.add_argument("--levels", type = str, default = "0,1,2,3")
    ap.add_argument("--no_vis", action = "store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok = True)
    os.makedirs(args.save_vis, exist_ok = True)
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip() != "")
    W_eval, H_eval = args.img_w, args.img_h

    device = get_device()
    # data
    if args.mode == "t":
        ds = RGBTCC_TDataset(args.data_root, args.split, (H_eval, W_eval))
    else:
        ds = RGBTCC_PairedDataset(args.data_root, args.split, (H_eval, W_eval))

    # model
    if args.mode == "t":
        model = CSRNet(load_imagenet = False).to(device)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = False).to(device)
    else:
        model = CSRNetRGBT_Late(load_imagenet = False).to(device)

    ckpt_path = os.path.expanduser(args.ckpt)
    if os.path.isdir(ckpt_path):
        cands = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith((".pth", ".pt"))]
        if not cands: raise FileNotFoundError(f"No .pth in {ckpt_path}")
        ckpt_path = max(cands, key = os.path.getmtime)
        print(f"[info] Using newest checkpoint: {ckpt_path}")
    elif not os.path.isfile(ckpt_path):
        alt = os.path.join(ckpt_path, "best.pth")
        if os.path.isfile(alt):
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    load_state_flex(model, ckpt_path, map_location = device)
    model.eval()

    header = ["image_id", "gt_count", "pred_count", "abs_err"] + [f"GAME{L}" for L in levels]
    rows = [tuple(header)]
    N = len(ds); mae = 0.0; rmse = 0.0; game_acc = np.zeros(len(levels), dtype = np.float64)

    with torch.no_grad():
        for sample in ds:
            if args.mode == "t":
                x_t3, den_gt, name, gt = sample
                x_t3 = x_t3.unsqueeze(0).to(device)
                den_pred = model(x_t3).squeeze().cpu().numpy()
                sid = os.path.splitext(name)[0]
                rgb_path = os.path.join(ds.split_dir, f"{sid}_RGB.jpg")
                if not os.path.exists(rgb_path):
                    rgb_path = os.path.join(ds.split_dir, f"{sid}_RGB.png")
            elif args.mode == "early":
                x_rgb, x_t3, den_gt, name, gt = sample
                t_gray = x_t3[:1, :, :]
                x4 = torch.cat([x_rgb, t_gray], dim = 0).unsqueeze(0).to(device)
                den_pred = model(x4).squeeze().cpu().numpy()
                sid = os.path.splitext(name)[0]
                rgb_path = os.path.join(ds.split_dir, f"{sid}_RGB.jpg")
                if not os.path.exists(rgb_path):
                    rgb_path = os.path.join(ds.split_dir, f"{sid}_RGB.png")
            else:
                x_rgb, x_t3, den_gt, name, gt = sample
                den_pred = model(x_rgb.unsqueeze(0).to(device), x_t3.unsqueeze(0).to(device)).squeeze().cpu().numpy()
                sid = os.path.splitext(name)[0]
                rgb_path = os.path.join(ds.split_dir, f"{sid}_RGB.jpg")
                if not os.path.exists(rgb_path):
                    rgb_path = os.path.join(ds.split_dir, f"{sid}_RGB.png")

            c_pred = float(den_pred.sum())
            err = abs(c_pred - gt)
            mae += err
            rmse += (c_pred - gt) ** 2

            den_up = cv2.resize(den_pred, (W_eval, H_eval), interpolation = cv2.INTER_CUBIC)
            orig = cv2.imread(rgb_path); H0, W0 = orig.shape[:2]
            gt_no_ext = os.path.join(ds.split_dir, f"{sid}_GT")
            pts = load_points(gt_no_ext)
            if pts.size > 0:
                pts = pts.copy()
                pts[:, 0] *= (W_eval / float(W0))
                pts[:, 1] *= (H_eval / float(H0))
            game_list = game_per_image(den_up, pts, W_eval, H_eval, levels)
            game_acc += np.array(game_list, dtype = np.float64)

            if not args.no_vis:
                m = den_up.max()
                dm_vis = np.zeros_like(den_up, dtype = np.uint8) if m <= 1e-6 else (den_up / m * 255).astype(np.uint8)
                heat = cv2.applyColorMap(dm_vis, cv2.COLORMAP_JET)
                # build an overlay base (use RGB image for consistency)
                img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                img_res = cv2.resize(img_rgb, (W_eval, H_eval), interpolation = cv2.INTER_LINEAR)
                overlay = cv2.addWeighted(cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR), 0.6, heat, 0.4, 0.0)
                cv2.putText(overlay, f"GT: {gt:.0f}  Pred: {c_pred:.1f}", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(args.save_vis, f"vis_{sid}.png"), overlay)

            rows.append((name, f"{gt:.1f}", f"{c_pred:.2f}", f"{err:.2f}", *[f"{g:.2f}" for g in game_list]))

    mae /= N
    rmse = math.sqrt(rmse / N)
    game_avg = (game_acc / N).tolist()
    print(f"[{args.mode}|{args.split}] MAE={mae:.2f} RMSE={rmse:.2f} " +
          " ".join([f"GAME{L}={g:.2f}" for L, g in zip(levels, game_avg)]) + f" N={N}")

    rows.append(("AVERAGE", f"{mae:.2f}", f"{rmse:.2f}", f"{mae:.2f}", *[f"{g:.2f}" for g in game_avg]))
    with open(args.save_csv, "w", newline = "") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved CSV to {args.save_csv} and overlays to {args.save_vis}")

if __name__ == "__main__":
    main()