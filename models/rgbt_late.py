import torch
import torch.nn as nn
from models.csrnet import CSRNet

class CSRNetRGBT_Late(nn.Module):
    def __init__(self, load_imagenet = True):
        super().__init__()
        self.rgb_net = CSRNet(load_imagenet = load_imagenet)
        self.t_net   = CSRNet(load_imagenet = load_imagenet)
        self.fuse    = nn.Conv2d(2, 1, kernel_size = 1, bias = True)

        nn.init.kaiming_normal_(self.fuse.weight, mode = 'fan_out', nonlinearity = 'relu')
        if self.fuse.bias is not None:
            nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x_rgb, x_t3):
        den_rgb = self.rgb_net(x_rgb)
        den_t   = self.t_net(x_t3)
        x = torch.cat([den_rgb, den_t], dim = 1)   # [B, 2, H/8, W/8]
        out = self.fuse(x)
        return out