import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


class _GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(16, 1, kernel_size = 1, padding = 0, bias = True)
        self.act = nn.ReLU(inplace = True)
        self.sig = nn.Sigmoid()

        # make gate start at 0.5 everywhere: sigmoid(0) = 0.5
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, pred_rgb, pred_t):
        x = torch.cat([pred_rgb, pred_t], dim = 1)  # (B,2,H,W)
        x = self.act(self.conv1(x))
        g = self.sig(self.conv2(x))
        return g


class CSRNetRGBT_AdaptiveLate(nn.Module):
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

        g = self.gate(pred_rgb, pred_t)  # (B,1,h,w)
        return g * pred_rgb + (1.0 - g) * pred_t