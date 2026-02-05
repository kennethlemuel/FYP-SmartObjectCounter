import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


def _resize_density_sum_preserving(den, size_hw):
    """
    Resize a density map to (H, W) while preserving total sum (count).
    den: (B, 1, H, W)
    size_hw: (H_new, W_new)
    """
    old_h, old_w = den.shape[-2], den.shape[-1]
    new_h, new_w = int(size_hw[0]), int(size_hw[1])

    if old_h == new_h and old_w == new_w:
        return den

    in_sum = den.sum(dim = (2, 3), keepdim = True)
    den_rs = F.interpolate(den, size = (new_h, new_w), mode = "bilinear", align_corners = False)
    out_sum = den_rs.sum(dim = (2, 3), keepdim = True).clamp_min(1e-6)
    return den_rs * (in_sum / out_sum)


class CSRNetRGBT_Late(nn.Module):
    """
    Late fusion baseline:
    - Two CSRNet experts (RGB and T).
    - Fuse their density maps with a 1x1 conv.
    """
    def __init__(self, load_imagenet = True):
        super().__init__()
        self.rgb_net = CSRNet(load_imagenet = load_imagenet)
        self.t_net = CSRNet(load_imagenet = load_imagenet)
        self.fuse = nn.Conv2d(2, 1, kernel_size = 1, bias = True)

        nn.init.kaiming_normal_(self.fuse.weight, mode = "fan_out", nonlinearity = "relu")
        if self.fuse.bias is not None:
            nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x_rgb, x_t3):
        den_rgb = self.rgb_net(x_rgb)
        den_t = self.t_net(x_t3)

        if den_t.shape[-2:] != den_rgb.shape[-2:]:
            den_t = _resize_density_sum_preserving(den_t, den_rgb.shape[-2:])

        x = torch.cat([den_rgb, den_t], dim = 1)  # [B, 2, H', W']
        out = self.fuse(x)
        return torch.relu(out)
