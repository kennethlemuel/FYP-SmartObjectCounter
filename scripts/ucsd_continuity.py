import os, argparse, random, csv
import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from models.csrnet import CSRNet

def smooth_counts(counts, k = 5, delta = 2):
    sm = []
    for i, c in enumerate(counts):
        lo = max(0, i-k+1)
        med = np.median(counts[lo:i+1])
        if i==0:
            sm.append(med)
        else:
            sm.append(np.clip(med, sm[-1] - delta, sm[-1] + delta))
    return np.array(sm, dtype=np.float32)

def gate_events(flow_band, mag_th = 1.0, min_gap = 8):
    mags = (flow_band[...,1]**2 + flow_band[...,0]**2)**0.5
    s = mags.mean(axis = 1).mean()
    return s > mag_th

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required = True)
    ap.add_argument("--ckpt", required = True)
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--median_k", type = int, default = 5)
    ap.add_argument("--delta_clamp", type = int, default = 2)
    ap.add_argument("--drop_rate", type = float, default = 0.10)
    ap.add_argument("--out_raw", required = True)
    ap.add_argument("--out_smooth", required = True)
    ap.add_argument("--plot", required = True)
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.jpg','.png'))])
    paths = [os.path.join(args.frames_dir, f) for f in files]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(load_imagenet = False).to(device)
    state = torch.load(args.ckpt, map_location = device)
    model.load_state_dict(state["model"]); model.eval()

    tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),])

    counts = []
    prev_gray = None
    hband_y = int(0.6 * args.img_h)
    band_h = max(8, args.img_h // 32)

    with torch.no_grad():
        for p in paths:
            img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
            inp = tf(img_res).unsqueeze(0).to(device)
            den = model(inp).squeeze().cpu().numpy()
            c = max(0.0, float(den.sum()))
            counts.append(c)

    counts = np.array(counts, dtype=np.float32)
    counts = np.maximum(counts, 0.0)
    counts_sm = smooth_counts(counts, k = args.median_k, delta = args.delta_clamp)

    with open(args.out_raw, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame","count_raw"])
        for i, c in enumerate(counts): w.writerow([i,int(c * 100) / 100])

    with open(args.out_smooth, "w", newline = "") as f:
        w = csv.writer(f); w.writerow(["frame", "count_smooth"])
        for i, c in enumerate(counts_sm): w.writerow([i,int(c * 100)/ 100])

    keep_mask = np.random.rand(len(counts)) > args.drop_rate
    dropped_counts = counts[keep_mask]
    dropped_counts_sm = smooth_counts(dropped_counts, k = args.median_k, delta = args.delta_clamp)

    plt.figure(figsize = (10, 4))
    plt.plot(counts, label = "raw")
    plt.plot(counts_sm, label = "continuity")
    plt.title("UCSD: per-frame counts (raw vs continuity)")
    plt.xlabel("frame") 
    plt.ylabel("count")
    plt.legend() 
    plt.tight_layout()
    plt.savefig(args.plot, dpi = 160)
    print(f"Saved CSVs and plot to {os.path.dirname(args.plot)}")

if __name__ == "__main__":
    main()
