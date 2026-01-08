import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


def _resize_density_sum_preserving(den, size_hw):
    """
    Resize a density map to (H, W) while preserving the total sum (count).
    den: (B, 1, H, W)
    size_hw: (H_new, W_new)
    """
    old_h, old_w = den.shape[-2], den.shape[-1]
    new_h, new_w = int(size_hw[0]), int(size_hw[1])

    if old_h == new_h and old_w == new_w:
        return den

    den_rs = F.interpolate(den, size = (new_h, new_w), mode = "bilinear", align_corners = False)
    den_rs = den_rs * (old_h * old_w) / float(new_h * new_w)
    return den_rs


class _GateNet(nn.Module):
    """
    Lightweight per-pixel gate that fuses two density predictions.
    Starts at 0.5 everywhere (equal weighting) for stable early training.
    """
    def __init__(self, hidden = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(4, hidden, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size = 1, padding = 0, bias = True)
        self.act = nn.ReLU(inplace = True)
        self.sig = nn.Sigmoid()

        # Start with g = 0.5 everywhere: sigmoid(0) = 0.5
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, pred_rgb, pred_t):
        x = torch.cat(
            [
                pred_rgb,
                pred_t,
                torch.abs(pred_rgb - pred_t),
                0.5 * (pred_rgb + pred_t),
            ],
            dim = 1
        )
        x = self.act(self.conv1(x))
        g = self.sig(self.conv2(x))
        return g


class CSRNetRGBT_AdaptiveLate(nn.Module):
    """
    Two CSRNet experts (RGB and T) with an adaptive per-pixel late-fusion gate.
    """
    def __init__(self, load_imagenet = True):
        super().__init__()
        self.rgb_net = CSRNet(load_imagenet = load_imagenet)
        self.t_net = CSRNet(load_imagenet = load_imagenet)
        self.gate = _GateNet()

    def forward(self, x_rgb, x_t3):
        pred_rgb = self.rgb_net(x_rgb)  # (B,1,h,w)
        pred_t = self.t_net(x_t3)       # (B,1,h,w)

        if pred_t.shape[-2:] != pred_rgb.shape[-2:]:
            pred_t = _resize_density_sum_preserving(pred_t, pred_rgb.shape[-2:])

        g = self.gate(pred_rgb, pred_t)  # (B,1,h,w) in [0,1]
        return g * pred_rgb + (1.0 - g) * pred_t