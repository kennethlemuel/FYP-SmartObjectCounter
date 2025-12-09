import os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from models.rgbt_early import CSRNetRGBT_Early
from models.rgbt_late import CSRNetRGBT_Late
from datasets.rgbt_cc import RGBTCC_TDataset, RGBTCC_PairedDataset


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
    ap.add_argument("--mode", choices = ["t", "early", "late"], required = True)
    ap.add_argument("--data_root", required = True)
    ap.add_argument("--split_train", default = "train")
    ap.add_argument("--split_val",   default = "val")
    ap.add_argument("--img_h", type = int, default = 640)
    ap.add_argument("--img_w", type = int, default = 960)
    ap.add_argument("--epochs", type = int, default = 10)
    ap.add_argument("--batch_size", type = int, default = 2)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--save_dir", default = "outputs/rgbt_baselines")
    ap.add_argument("--num_workers", type = int, default = -1, help = "-1 = auto (0 on CPU/MPS, 4 on CUDA)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    device = get_device()
    print(f"[init] device = {device}")

    #datasets
    if args.mode == "t":
        train_ds = RGBTCC_TDataset(args.data_root, args.split_train, (args.img_h, args.img_w))
        val_ds   = RGBTCC_TDataset(args.data_root, args.split_val,   (args.img_h, args.img_w))
    else:
        train_ds = RGBTCC_PairedDataset(args.data_root, args.split_train, (args.img_h, args.img_w))
        val_ds   = RGBTCC_PairedDataset(args.data_root, args.split_val,   (args.img_h, args.img_w))

    if args.num_workers < 0:
        workers = 4 if device.type == "cuda" else 0
    else:
        workers = args.num_workers
    pin = (device.type == "cuda") and (workers > 0)

    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True,
                          num_workers = workers, pin_memory = pin)
    val_dl   = DataLoader(val_ds, batch_size = 1, shuffle = False,
                          num_workers = workers, pin_memory = pin)

    print(f"[init] train = {len(train_ds)}  val = {len(val_ds)}  workers = {workers}")

    #models
    if args.mode == "t":
        model = CSRNet(load_imagenet = True).to(device)
    elif args.mode == "early":
        model = CSRNetRGBT_Early(load_imagenet = True).to(device)
    else:
        model = CSRNetRGBT_Late(load_imagenet = True).to(device)

    optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    mse   = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled = (args.amp and device.type == "cuda"))

    best_mae = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0

        for step, batch in enumerate(train_dl, 1):
            if args.mode == "t":
                x_t3, den, _, _ = batch
                x_t3 = x_t3.to(device, non_blocking = True)
                den  = den.to(device,  non_blocking = True)
            else:
                x_rgb, x_t3, den, _, _ = batch
                x_rgb = x_rgb.to(device, non_blocking = True)
                x_t3  = x_t3.to(device,  non_blocking = True)
                den   = den.to(device,   non_blocking = True)

            optim.zero_grad(set_to_none = True)
            with torch.amp.autocast(device_type = _autocast_device_type(device),
                                    enabled = (args.amp and device.type == "cuda")):
                if args.mode == "t":
                    pred = model(x_t3)
                elif args.mode == "early":
                    #here, i build 4-channel input: 3 RGB + 1 T_gray
                    t_gray = x_t3[:, :1, :, :]          #the thermal replicated by using any 1 channel
                    x4 = torch.cat([x_rgb, t_gray], dim = 1)
                    pred = model(x4)
                else:
                    pred = model(x_rgb, x_t3)
                # ensure target matches pred spatial size
                if pred.shape[-2:] != den.shape[-2:]:
                    den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)
                loss = mse(pred, den)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            run_loss += float(loss.item())

            if step == 1 or step % 50 == 0:
                print(f"[e{ep:02d} s{step:04d}/{len(train_dl)}] loss = {loss.item():.4f}")

        # validation
        model.eval()
        mae = rmse = 0.0
        with torch.no_grad():
            for batch in val_dl:
                if args.mode == "t":
                    x_t3, den, _, _ = batch
                    x_t3 = x_t3.to(device)
                    pred = model(x_t3)
                elif args.mode == "early":
                    x_rgb, x_t3, den, _, _ = batch
                    t_gray = x_t3[:, :1, :, :]
                    x4 = torch.cat([x_rgb.to(device), t_gray.to(device)], dim = 1)
                    pred = model(x4)
                else:
                    x_rgb, x_t3, den, _, _ = batch
                    pred = model(x_rgb.to(device), x_t3.to(device))

                den = den.to(device)
                if pred.shape[-2:] != den.shape[-2:]:
                    den = F.interpolate(den, size = pred.shape[-2:], mode = "bilinear", align_corners = False)

                c_pred = float(pred.sum().item())
                c_gt   = float(den.sum().item())
                mae   += abs(c_pred - c_gt)
                rmse  += (c_pred - c_gt) ** 2

        mae  /= len(val_dl)
        rmse  = math.sqrt(rmse / len(val_dl))
        print(f"Epoch {ep:02d}: train_loss = {run_loss / max(1, len(train_dl)):.4f}  MAE = {mae:.2f}  RMSE = {rmse:.2f}")

        if mae < best_mae:
            best_mae = mae
            tag = f"{args.mode}_best.pth"
            torch.save({"epoch": ep, "model": model.state_dict()}, os.path.join(args.save_dir, tag))
            print(f"-> saved {tag} (MAE {best_mae:.2f})")

if __name__ == "__main__":
    main()