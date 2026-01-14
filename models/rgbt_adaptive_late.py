import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


def _resize_density_sum_preserving(x: torch.Tensor, size_hw: tuple[int, int], eps: float = 1e-6) -> torch.Tensor:
    """
    Resize a density map while (approximately) preserving total count (sum).
    Uses bilinear resize, then rescales to match original sum per-sample.
    Runs in float32 for stability.
    """
    if x.shape[-2:] == size_hw:
        return x

    x32 = x.float()
    old_sum = x32.sum(dim = (-2, -1), keepdim = True)  # [B,1,1,1] or [B,C,1,1]

    y = F.interpolate(x32, size = size_hw, mode = "bilinear", align_corners = False)
    new_sum = y.sum(dim = (-2, -1), keepdim = True)

    scale = old_sum / (new_sum + eps)
    y = y * scale
    return y.to(dtype = x.dtype)


class CSRNetRGBT_AdaptiveLate(nn.Module):
    """
    Adaptive late fusion:
      - Two CSRNet branches (RGB, Thermal replicated to 3ch)
      - Gate predicted from concatenated density predictions
      - Output: gate * den_rgb + (1 - gate) * den_t

    Key stability choices:
      - Gate + fusion computed in float32
      - Gate clamped to (eps, 1-eps)
      - If branch outputs mismatch in spatial size, resize density sum-preserving
    """

    def __init__(self, load_imagenet: bool = True, gate_hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.rgb_net = CSRNet(load_imagenet = load_imagenet)
        self.t_net = CSRNet(load_imagenet = load_imagenet)
        self.eps = float(eps)

        self.gate = nn.Sequential(
            nn.Conv2d(2, gate_hidden, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(gate_hidden, 1, kernel_size = 1, bias = True),
        )

        # Init gate convs for stable start
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x_rgb: torch.Tensor, x_t3: torch.Tensor, return_gate: bool = False):
        den_rgb = self.rgb_net(x_rgb)   # [B,1,h,w]
        den_t = self.t_net(x_t3)        # [B,1,h,w]

        # Align spatial sizes if needed (rare, but keeps it robust)
        if den_t.shape[-2:] != den_rgb.shape[-2:]:
            den_t = _resize_density_sum_preserving(den_t, den_rgb.shape[-2:], eps = self.eps)

        # Compute gate + fusion in float32 for AMP stability
        den_rgb32 = den_rgb.float()
        den_t32 = den_t.float()

        gate_logits = self.gate(torch.cat([den_rgb32, den_t32], dim = 1))
        gate = torch.sigmoid(gate_logits)

        # Prevent exact 0/1 which can break any log-based regularizer downstream
        gate = gate.clamp(self.eps, 1.0 - self.eps)

        out32 = gate * den_rgb32 + (1.0 - gate) * den_t32
        out = out32.to(dtype = den_rgb.dtype)

        if return_gate:
            return out, gate
        return out
