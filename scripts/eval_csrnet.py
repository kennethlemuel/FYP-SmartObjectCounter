# scripts/eval_csrnet.py
import os, csv, argparse, math
import numpy as np
import cv2
import torch
from torchvision import transforms
from scipy.io import loadmat
from models.csrnet import CSRNet

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def load_dataset_paths(root):
    img_dir = os.path.join(root, "images")
    gt_dir  = os.path.join(root, "ground_truth")
    imgs = sorted([f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_list = [f for i, f in enumerate(imgs) if i % 5 == 0]  # ShanghaiTech-B style
    return img_dir, gt_dir, val_list

def load_state_flex(model, ckpt_path, map_location):
    state = torch.load(ckpt_path, map_location = map_location)
    if isinstance(state, dict):
        for k in ["model", "state_dict", "ema", "net", "model_state_dict"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    model.load_state_dict(state, strict = False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--ckpt", required = True, help = "path to .pth or a folder containing checkpoints")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--save_csv", required = True)
    ap.add_argument("--save_vis", required = True)
    args = ap.parse_args()

    os.makedirs(args.save_vis, exist_ok = True)
    os.makedirs(os.path.dirname(args.save_csv), exist_ok = True)

    device = get_device()
    model = CSRNet(load_imagenet = False).to(device)

    # allow directory or file; also try best.pth
    ckpt_path = os.path.expanduser(args.ckpt)
    if os.path.isdir(ckpt_path):
        cands = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path)
                 if f.endswith((".pth", ".pt"))]
        if not cands:
            raise FileNotFoundError(f"No .pth/.pt in directory: {ckpt_path}")
        ckpt_path = max(cands, key = os.path.getmtime)
        print(f"[info] Using newest checkpoint: {ckpt_path}")
    elif not os.path.isfile(ckpt_path):
        alt = os.path.join(ckpt_path, "best.pth")
        if os.path.isfile(alt):
            ckpt_path = alt
            print(f"[info] Using {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    load_state_flex(model, ckpt_path, map_location = device)
    model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
    ])

    img_dir, gt_dir, files = load_dataset_paths(args.data_root)

    rows = [("image_id", "gt_count", "pred_count", "abs_err")]
    mae = 0.0
    rmse = 0.0
    n = 0

    for name in files:
        img_path = os.path.join(img_dir, name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[warn] cannot read image: {name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        img_res = cv2.resize(img_rgb, (args.img_w, args.img_h), interpolation = cv2.INTER_LINEAR)

        gt_mat_path = os.path.join(gt_dir, f"GT_{os.path.splitext(name)[0]}.mat")
        mat = loadmat(gt_mat_path)
        pts = np.array(mat["image_info"][0, 0][0, 0][0], dtype = np.float32)
        c_gt = float(len(pts))

        with torch.no_grad():
            inp = tf(img_res).unsqueeze(0).to(device)
            den_pred = model(inp).squeeze().cpu().numpy()

        c_pred = float(den_pred.sum())
        err = abs(c_pred - c_gt)
        rows.append((name, f"{c_gt:.1f}", f"{c_pred:.2f}", f"{err:.2f}"))

        mae += err
        rmse += (c_pred - c_gt) ** 2
        n += 1

        # overlay
        dm_up = cv2.resize(den_pred, (args.img_w, args.img_h), interpolation = cv2.INTER_CUBIC)
        m = dm_up.max()
        dm_vis = np.zeros_like(dm_up, dtype = np.uint8) if m <= 1e-6 else (dm_up / m * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(dm_vis, cv2.COLORMAP_JET)
        base_bgr = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.6, heat, 0.4, 0.0)
        cv2.putText(overlay, f"GT: {c_gt:.0f}  Pred: {c_pred:.1f}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(args.save_vis, f"vis_{os.path.splitext(name)[0]}.png")
        cv2.imwrite(out_path, overlay)

    if n > 0:
        mae /= n
        rmse = math.sqrt(rmse / n)
        rows.append(("AVERAGE", f"{mae:.2f}", "", f"{mae:.2f}"))

    with open(args.save_csv, "w", newline = "") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved {len(rows) - 1} results to {args.save_csv} and overlays to {args.save_vis}")

if __name__ == "__main__":
    main()