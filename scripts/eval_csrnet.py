import os, csv, argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.io import loadmat
from models.csrnet import CSRNet

OUT_SRIDE = 8

def load_dataset_paths(root):
    img_dir = os.path.join(root, "images")
    gt_dir = os.path.join(root, "ground_truth")
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_list = [f for i, f in enumerate(imgs) if i%5 == 0]
    return img_dir, gt_dir, val_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--img_h", type = int, default = True)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--save_csv", required = True)
    ap.add_argument("--save_vis", required = True)
    args = ap.parse_args()

    os.makedirs(args.save_vis, exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(load_imagenet = False).to(device)
    state = torch.load(args.ckpt, map_location = device)
    model.load_state_dict(state["model"])
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])

    img_dir, gt_dir, files = load_dataset_paths(args.data_root)

    rows = [("image_id", "gt_count", "pred_count", "abs_err")]
    for name in files:

        img_bgr = cv2.imread(os.path.join(img_dir, name))
        if img_bgr is None:
            print(f"[warn] cannot read image: {name}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        img_res = cv2.resize(img_rgb, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)

        gt_mat_path = os.path.join(gt_dir, f"GT_{os.path.splitext(name)[0]}.mat")
        mat = loadmat(gt_mat_path)
        pts = np.array(mat['image_info'][0,0][0,0][0], dtype=np.float32)
        c_gt = float(len(pts))

        with torch.no_grad():
            inp = tf(img_res).unsqueeze(0).to(device)
            den_pred = model(inp).squeeze(0).squeeze(0).cpu().numpy()
        c_pred = float(den_pred.sum())
        rows.append((name, f"{c_gt: .1f}", f"{c_pred: .2f}", f"{abs(c_pred - c_gt): .2f}"))

        dm_up = cv2.resize(den_pred, (args.img_w, args.img_h), interpolation = cv2.INTER_CUBIC)
        m = dm_up.max()
        dm_vis = np.zeros_like(dm_up, dtype=np.uint8) if m <= 1e-6 else (dm_up / m * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(dm_vis, cv2.COLORMAP_JET)
        base_bgr = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.6, heat, 0.4, 0.0)
        cv2.putText(overlay, f"GT: {c_gt: .0f}  Pred: {c_pred: .1f}",(12, 32), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(args.save_vis, f"vis_{os.path.splitext(name)[0]}.png")
        cv2.imwrite(out_path, overlay)

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, "w", newline = "") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved {len(rows) - 1} results to {args.save_csv} and overlays to {args.save_vis}")


if __name__ == "__main__":
    main()