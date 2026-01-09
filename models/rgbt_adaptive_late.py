import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _resize_density_sum_preserving(den: torch.Tensor, size_hw):
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


def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 3, H, W) normalized by ImageNet mean/std.
    Returns values roughly in [0,1] (clipped).
    """
    mean = _IMAGENET_MEAN.to(device = x.device, dtype = x.dtype)
    std = _IMAGENET_STD.to(device = x.device, dtype = x.dtype)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


def _luminance(rgb_01: torch.Tensor) -> torch.Tensor:
    """
    rgb_01: (B, 3, H, W) in [0,1]
    returns: (B, 1, H, W) in [0,1]
    """
    r = rgb_01[:, 0:1]
    g = rgb_01[:, 1:2]
    b = rgb_01[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clamp(0.0, 1.0)


class _GateNet(nn.Module):
    """
    Lightweight per-pixel gate that fuses two density predictions.

    Inputs:
      - pred_rgb (B,1,h,w)
      - pred_t   (B,1,h,w)
      - lum      (B,1,h,w)
      - t_int    (B,1,h,w)

    Output:
      - g (B,1,h,w) in [0,1], where:
          fused = g * pred_rgb + (1 - g) * pred_t
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(6, hidden, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size = 1, padding = 0, bias = True)
        self.act = nn.ReLU(inplace = True)
        self.sig = nn.Sigmoid()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, pred_rgb, pred_t, lum, t_int):
        x = torch.cat(
            [
                pred_rgb,
                pred_t,
                torch.abs(pred_rgb - pred_t),
                0.5 * (pred_rgb + pred_t),
                lum,
                t_int,
            ],
            dim = 1
        )
        x = self.act(self.conv1(x))
        g = self.sig(self.conv2(x))
        return g


class CSRNetRGBT_AdaptiveLate(nn.Module):
    """
    Adaptive late fusion:
    - Two CSRNet experts (RGB and T).
    - Per-pixel gating conditioned on predictions + illumination cues.

    Improvements:
    - Softplus on outputs to enforce non-negative densities.
    - 1x1 calibration per expert to learn small scale/bias corrections.
    """
    def __init__(self, load_imagenet: bool = True, gate_hidden: int = 32, softplus_beta: float = 1.0):
        super().__init__()
        self.rgb_net = CSRNet(load_imagenet = load_imagenet)
        self.t_net = CSRNet(load_imagenet = load_imagenet)

        self.rgb_cal = nn.Conv2d(1, 1, kernel_size = 1, bias = True)
        self.t_cal = nn.Conv2d(1, 1, kernel_size = 1, bias = True)
        nn.init.ones_(self.rgb_cal.weight)
        nn.init.zeros_(self.rgb_cal.bias)
        nn.init.ones_(self.t_cal.weight)
        nn.init.zeros_(self.t_cal.bias)

        self.pos = nn.Softplus(beta = float(softplus_beta))
        self.gate = _GateNet(hidden = gate_hidden)

    def forward(self, x_rgb, x_t3):
        pred, _ = self.forward_with_aux(x_rgb, x_t3)
        return pred

    @torch.no_grad()
    def _make_lum_and_tint(self, x_rgb, x_t3, out_hw):
        rgb_01 = _denormalize_imagenet(x_rgb)
        t3_01 = _denormalize_imagenet(x_t3)

        lum = _luminance(rgb_01)
        t_int = t3_01.mean(dim = 1, keepdim = True)

        lum = F.adaptive_avg_pool2d(lum, out_hw)
        t_int = F.adaptive_avg_pool2d(t_int, out_hw)

        return lum, t_int

    def forward_with_aux(self, x_rgb, x_t3):
        """
        Returns:
          pred_fused: (B,1,h,w)
          aux: dict with keys {"gate", "lum"} for optional regularization
        """
        pred_rgb = self.rgb_net(x_rgb)
        pred_t = self.t_net(x_t3)

        if pred_t.shape[-2:] != pred_rgb.shape[-2:]:
            pred_t = _resize_density_sum_preserving(pred_t, pred_rgb.shape[-2:])

        pred_rgb = self.pos(self.rgb_cal(pred_rgb))
        pred_t = self.pos(self.t_cal(pred_t))

        with torch.no_grad():
            lum, t_int = self._make_lum_and_tint(x_rgb, x_t3, pred_rgb.shape[-2:])

        g = self.gate(pred_rgb, pred_t, lum, t_int)
        pred_fused = g * pred_rgb + (1.0 - g) * pred_t

        aux = {
            "gate": g,
            "lum": lum,
        }
        return pred_fused, aux
