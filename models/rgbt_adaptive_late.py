import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


class _GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 1, kernel_size = 1, padding = 0, bias = True),
            nn.Sigmoid(),
        )

    def forward(self, pred_rgb, pred_t):
        x = torch.cat([pred_rgb, pred_t, torch.abs(pred_rgb - pred_t)], dim = 1)
        return self.net(x)


class CSRNetRGBT_AdaptiveLate(nn.Module):
    def __init__(self, load_imagenet = True):
        super().__init__()
        self.rgb_net = CSRNet(load_imagenet = load_imagenet)
        self.t_net = CSRNet(load_imagenet = load_imagenet)
        self.gate = _GateNet()

    def forward(self, x_rgb, x_t3):
        pred_rgb = self.rgb_net(x_rgb)
        pred_t = self.t_net(x_t3)

        if pred_rgb.shape[-2:] != pred_t.shape[-2:]:
            pred_t = F.interpolate(pred_t, size = pred_rgb.shape[-2:], mode = "bilinear", align_corners = False)

        g = self.gate(pred_rgb, pred_t)
        pred = g * pred_rgb + (1.0 - g) * pred_t
        return pred