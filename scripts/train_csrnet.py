import os, random, argparse, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from models.csrnet import CSRNet
import time

OUT_STRIDE = 8

def set_seed(s = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def density_from_points(points_xy, h, w, sigma = 15):
    dm = np.zeros((h,w), dtype = np.float32)
    if points_xy.size == 0:
        return dm
    xs = np.clip(points_xy[:,0].astype(int), 0, w-1)
    ys = np.clip(points_xy[:,1].astype(int), 0, h-1)
    dm[ys, xs] = 1.0
    dm = gaussian_filter(dm, sigma = sigma, mode = 'constant')
    s = dm.sum()
    if s > 0:
        dm *= (len(xs)/s)
    return dm

class SHTBDataset(Dataset):
    def __init__(self, root, split, img_size, sigma, max_count = None):
        self.root = root
        self.split = split
        self.h, self.w = img_size
        self.sigma = sigma

        img_dir = os.path.join(root, "images")
        gt_dir = os.path.join(root, "ground_truth")
        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_list = [f for i, f in enumerate(imgs) if i%5!=0]
        val_list = [f for i, f in enumerate(imgs) if i%5==0]

        self.files = train_list if split == "train" else val_list
        if max_count is not None:
            self.files = self.files[:max_count]

        self.img_dir, self.gt_dir = img_dir, gt_dir
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229, 0.224, 0.225])])

        self.h_out = self.h // OUT_STRIDE
        self.w_out = self.w // OUT_STRIDE
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        mat = loadmat(os.path.join(self.gt_dir, "GT_" + os.path.splitext(name)[0] + ".mat"))
        pts = mat['image_info'][0, 0][0, 0][0]
        pts = np.array(pts, dtype = np.float32)

        img_res = cv2.resize(img, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        if pts.size > 0:
            scale_x, scale_y = self.w / W, self.h / H
            pts[:,0] *= scale_x
            pts[:,1] *= scale_y

        pts_out = pts.copy()
        if pts_out.size > 0:
            pts_out[:,0] /= OUT_STRIDE
            pts_out[:,1] /= OUT_STRIDE
        sigma_out = max(1.0, self.sigma/OUT_STRIDE)
        den = density_from_points(pts_out, self.h_out, self.w_out, sigma = sigma_out)

        img_t = self.tf(img_res)
        den_t = torch.from_numpy(den).unsqueeze(0)
        count = float(den.sum())
        return img_t, den_t, name, count
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type = str, required = True)
    ap.add_argument("--img_h", type = int, default = 768)
    ap.add_argument("--img_w", type = int, default = 1024)
    ap.add_argument("--sigma", type = float, default = 15)
    ap.add_argument("--train_count", type = int, default = 500)
    ap.add_argument("--val_count", type = int, default = 100)
    ap.add_argument("--epochs", type = int, default = 15)
    ap.add_argument("--batch_size", type = int, default = 2)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--amp", action = "store_true")
    ap.add_argument("--save_dir", type = str, default = "outputs/csrnet_shtb")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(load_imagenet = True).to(device)

    train_ds = SHTBDataset(args.data_root, "train", (args.img_h, args.img_w), args.sigma, args.train_count)
    val_ds = SHTBDataset(args.data_root, "val", (args.img_h, args.img_w), args.sigma, args.val_count)

    # ////// Mac-safe DataLoader settings + immediate visibility
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=False)
    print(f"[init] train images={len(train_ds)}  val images={len(val_ds)}")
    print(f"[init] train batches={len(train_dl)}  val batches={len(val_dl)}")
    # //////

    optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled = args.amp)

    best_mae = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        #timing to check how the progress is going as it goes (used due to slow computer and checking required)
        t0 = time.time()
        for step, (img, den, _, _) in enumerate(train_dl, 1):
            img, den = img.to(device, non_blocking = True), den.to(device, non_blocking = True)
            optim.zero_grad(set_to_none = True)
            with torch.cuda.amp.autocast(enabled = args.amp):
                pred = model(img)
                loss = mse(pred,den)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            # timing is  every 20 steps (prints avg sec/batch)
            if step % 20 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"[epoch {ep:02d}] step {step}/{len(train_dl)} "
                      f"~{elapsed/step:.3f}s/batch  loss={loss.item():.4f}")
        train_loss = running/max(1, len(train_dl))
        model.eval()
        mae, rmse = 0.0, 0.0
        with torch.no_grad():
            for img, den, _,_ in val_dl:
                img = img.to(device)
                pred = model(img)
                c_pred = float(pred.sum().item())
                c_gt = float(den.sum().item())
                mae += abs(c_pred - c_gt)
                rmse += (c_pred - c_gt)**2
        mae /= len(val_dl)
        rmse = math.sqrt(rmse/len(val_dl))

        print(f"Epoch {ep: 02d}: train_loss = {train_loss: .4f} MAE = {mae: .2f} RMSE = {rmse: .2f}")
        if mae < best_mae:
            best_mae = mae
            torch.save({"epoch": ep, "model": model.state_dict()}, os.path.join(args.save_dir, "best.pth"))
            print(f"-> saved best.pth (MAE {best_mae: .2f})")

if __name__ == "__main__":
    main()