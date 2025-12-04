import os, argparse, math, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from datasets.rgbt_cc import RGBTCC_RGBDataset

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="data/RGBT-CC-CVPR2021")
    ap.add_argument("--img_h", type=int, default=768)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)   # 0 for mac; >0 on NSCC
    ap.add_argument("--save_dir", type=str, default="outputs/rgbtcc_rgb")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(load_imagenet=True).to(device)

    #the dataset are the train/val folders in RGBTCC
    train_ds = RGBTCC_RGBDataset(args.data_root, "train", (args.img_h, args.img_w), args.sigma)
    val_ds   = RGBTCC_RGBDataset(args.data_root, "val",   (args.img_h, args.img_w), args.sigma)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(args.num_workers>0))
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=(args.num_workers>0))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse   = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_mae = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()
        for step, (img, den, _, _) in enumerate(train_dl, 1):
            img, den = img.to(device, non_blocking=True), den.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(img)
                loss = mse(pred, den)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            run_loss += loss.item()

        train_loss = run_loss / max(1, len(train_dl))

        #validation (MAE/RMSE of counts)
        model.eval()
        mae, rmse = 0.0, 0.0
        with torch.no_grad():
            for img, den, _, _ in val_dl:
                img = img.to(device)
                pred = model(img)
                c_pred = float(pred.sum().item())
                c_gt   = float(den.sum().item())
                mae   += abs(c_pred - c_gt)
                rmse  += (c_pred - c_gt) ** 2
        mae  /= len(val_dl)
        rmse  = math.sqrt(rmse / len(val_dl))

        print(f"Epoch {ep:02d}: train_loss={train_loss:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"[{(time.time()-t0):.1f}s]")
        if mae < best_mae:
            best_mae = mae
            torch.save({"epoch": ep, "model": model.state_dict()},
                       os.path.join(args.save_dir, "best_rgb.pth"))
            print(f"-> saved best_rgb.pth (MAE {best_mae:.2f})")

if __name__ == "__main__":
    main()