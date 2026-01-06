import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


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
        # (B,1,H,W) -> (B,4,H,W)
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
            pred_t = F.interpolate(pred_t, size = pred_rgb.shape[-2:], mode = "bilinear", align_corners = False)

        g = self.gate(pred_rgb, pred_t)  # (B,1,h,w) in [0,1]
        return g * pred_rgb + (1.0 - g) * pred_t