import os, argparse, math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from datasets.rgbt_cc import RGBTCC_RGBDataset

def set_seed(s = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def _autocast_device_type(device):
    return "cuda" if device.type == "cuda" else "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required = True, help = "data/RGBT-CC-CVPR2021")
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15.0)
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--batch_size", type = int, default = 2)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--weight_decay", type = float, default = 1e-4)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--num_workers", type = int, default = 0)   # 0 for mac; >0 on NSCC
    ap.add_argument("--save_dir", type = str, default = "outputs/rgbtcc_rgb")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(42)

    device = get_device()
    print(f"[init] device = {device}")

    model = CSRNet(load_imagenet = True).to(device)

    # train/val splits from RGBT-CC
    train_ds = RGBTCC_RGBDataset(args.data_root, "train", (args.img_h, args.img_w), args.sigma)
    val_ds   = RGBTCC_RGBDataset(args.data_root, "val",   (args.img_h, args.img_w), args.sigma)

    pin = (device.type == "cuda") and (args.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True,
                          num_workers = args.num_workers, pin_memory = pin)
    val_dl   = DataLoader(val_ds, batch_size = 1, shuffle = False,
                          num_workers = args.num_workers, pin_memory = pin)

    print(f"[init] train images = {len(train_ds)}  val images = {len(val_ds)}")

    optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    mse   = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled = (args.amp and device.type == "cuda"))

    best_mae = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        for step, (img, den, _, _) in enumerate(train_dl, 1):
            img, den = img.to(device, non_blocking = True), den.to(device, non_blocking = True)
            optim.zero_grad(set_to_none = True)
            with torch.amp.autocast(device_type = _autocast_device_type(device),
                                    enabled = (args.amp and device.type == "cuda")):
                pred = model(img)
                # safety: match targets to prediction spatial size if needed
                if pred.shape[-2:] != den.shape[-2:]:
                    den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)
                loss = mse(pred, den)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            run_loss += loss.item()

            if step == 1 or step % 20 == 0:
                dt = time.time() - t0
                print(f"[e{ep:02d} s{step:04d}/{len(train_dl)}] loss={loss.item():.4f} ~{dt/step:.3f}s/batch")

        train_loss = run_loss / max(1, len(train_dl))

        # validation (MAE/RMSE of counts)
        model.eval()
        mae, rmse = 0.0, 0.0
        with torch.no_grad():
            for img, den, _, _ in val_dl:
                img, den = img.to(device, non_blocking = True), den.to(device, non_blocking = True)
                with torch.amp.autocast(device_type = _autocast_device_type(device),
                                        enabled = (args.amp and device.type == "cuda")):
                    pred = model(img)
                if pred.shape[-2:] != den.shape[-2:]:
                    den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)
                c_pred = float(pred.sum().item())
                c_gt   = float(den.sum().item())
                mae   += abs(c_pred - c_gt)
                rmse  += (c_pred - c_gt) ** 2

        mae  /= len(val_dl)
        rmse  = math.sqrt(rmse / len(val_dl))

        print(f"Epoch {ep:02d}: train_loss={train_loss:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  [{(time.time()-t0):.1f}s]")

        # always keep last; keep best on MAE
        torch.save({"epoch": ep, "model": model.state_dict()},
                   os.path.join(args.save_dir, "last_rgb.pth"))
        if mae < best_mae:
            best_mae = mae
            torch.save({"epoch": ep, "model": model.state_dict()},
                       os.path.join(args.save_dir, "best_rgb.pth"))
            print(f"-> saved best_rgb.pth (MAE {best_mae:.2f})")

if __name__ == "__main__":
    main()