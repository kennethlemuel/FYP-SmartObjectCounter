import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


def _resize_density_sum_preserving(d: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Resize a density map while approximately preserving total count (sum over H,W).
    d: [B,1,H,W]
    """
    if d.shape[-2:] == size:
        return d
    h0, w0 = d.shape[-2:]
    h1, w1 = size
    d_up = F.interpolate(d, size = size, mode = "bilinear", align_corners = False)
    # Preserve sum approximately: scale by area ratio
    scale = (h0 * w0) / float(h1 * w1)
    return d_up * scale


@torch.no_grad()
def _to_grayscale(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W]
    if x.shape[1] == 1:
        return x
    if x.shape[1] >= 3:
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    return x.mean(dim = 1, keepdim = True)


def _confidence_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Simple reliability proxy: normalized gradient magnitude in [0,1].
    Casts to float32 so ops like quantile work under AMP.
    Returns [B,1,H,W].
    """
    x = _to_grayscale(x.float())
    b, _, h, w = x.shape

    # Finite differences (padding to keep shape)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    mag = torch.sqrt(dx * dx + dy * dy + eps)

    # Robust normalize using top-1% quantile per image
    mag_flat = mag.view(b, -1)
    # quantile needs float/double; ensured by .float() above
    denom = (mag_flat.quantile(0.99, dim = 1) + eps).view(b, 1, 1, 1)
    conf = (mag / denom).clamp(0.0, 1.0)

    # Light smoothing helps stability
    conf = F.avg_pool2d(conf, kernel_size = 3, stride = 1, padding = 1)
    return conf


class CSRNetRGBT_AdaptiveLate(nn.Module):
    """
    Reliability-aware adaptive late fusion.

    - Two CSRNet experts (RGB, Thermal) predict density maps.
    - A gate network predicts a spatial mixing weight.
    - Confidence maps (from input gradients) modulate the gate so the fusion trusts the cleaner modality.

    This avoids temporal continuity and only needs paired RGB-T frames.
    """

    def __init__(self, load_imagenet: bool = True, load_weights: bool = True) -> None:
        super().__init__()
        # Accept both flags for compatibility with older scripts.
        use_pretrained = bool(load_imagenet or load_weights)

        self.rgb_net = CSRNet(load_imagenet = use_pretrained)
        self.t_net = CSRNet(load_imagenet = use_pretrained)

        # Gate network consumes [pred_rgb, pred_t, conf_rgb, conf_t] => 4 channels
        self.gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 1, kernel_size = 1),
        )

    def forward(self, x_rgb: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        # Make sure both modalities share input spatial size.
        if x_t.shape[-2:] != x_rgb.shape[-2:]:
            x_t = F.interpolate(x_t, size = x_rgb.shape[-2:], mode = "bilinear", align_corners = False)

        # Experts
        pred_rgb = self.rgb_net(x_rgb)
        pred_t = self.t_net(x_t)

        # Align prediction maps if any stride/padding mismatch occurs
        target_hw = pred_rgb.shape[-2:]
        if pred_t.shape[-2:] != target_hw:
            pred_t = _resize_density_sum_preserving(pred_t, target_hw)

        # Confidence maps (compute in float32; downsample to prediction resolution)
        conf_rgb = _confidence_map(x_rgb)
        conf_t = _confidence_map(x_t)

        conf_rgb = F.interpolate(conf_rgb, size = target_hw, mode = "bilinear", align_corners = False)
        conf_t = F.interpolate(conf_t, size = target_hw, mode = "bilinear", align_corners = False)

        # Gate
        gate_in = torch.cat([pred_rgb.float(), pred_t.float(), conf_rgb, conf_t], dim = 1)  # [B,4,H,W]
        gate = torch.sigmoid(self.gate(gate_in))  # [B,1,H,W] in [0,1]

        # Reliability-aware weighting (explicitly uses confidence)
        w_rgb = gate * conf_rgb
        w_t = (1.0 - gate) * conf_t
        denom = (w_rgb + w_t).clamp_min(1e-6)

        fused = (w_rgb / denom) * pred_rgb + (w_t / denom) * pred_t
        return fused
