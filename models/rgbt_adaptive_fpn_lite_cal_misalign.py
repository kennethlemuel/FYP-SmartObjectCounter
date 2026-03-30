import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgbt_adaptive_fpn_lite_cal import CSRNetRGBT_AdaptiveFPNLiteCal


class CSRNetRGBT_AdaptiveFPNLiteCalMisalign(CSRNetRGBT_AdaptiveFPNLiteCal):
    """
    Amendment Version 3:
    Adaptive FPN Lite + Calibration with explicit misalignment awareness.

    The model predicts:
    - a small global thermal shift to pre-align the thermal channel
    - a thermal confidence scalar that down-weights thermal contribution when
      RGB-T agreement is likely weak
    """

    def __init__(
        self,
        load_imagenet: bool = True,
        feat_ch: int = 64,
        scale_bound: float = 0.1,
        max_shift_px: float = 4.0,
        thermal_conf_floor: float = 0.25,
    ):
        super().__init__(load_imagenet = load_imagenet, feat_ch = feat_ch, scale_bound = scale_bound)
        self.max_shift_px = float(max_shift_px)
        self.thermal_conf_floor = float(thermal_conf_floor)

        self.align_feat = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.shift_head = nn.Linear(16, 2, bias = True)
        self.conf_head = nn.Linear(16, 1, bias = True)

        for m in self.align_feat.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, 2.0)

    def _encode_alignment(self, x4: torch.Tensor):
        feat = self.align_feat(x4).flatten(1)
        shift_px = torch.tanh(self.shift_head(feat)) * self.max_shift_px
        conf = torch.sigmoid(self.conf_head(feat))
        conf = self.thermal_conf_floor + (1.0 - self.thermal_conf_floor) * conf
        return shift_px, conf

    def _warp_thermal(self, thermal: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
        b, _, h, w = thermal.shape
        theta = thermal.new_zeros((b, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0

        if w > 1:
            theta[:, 0, 2] = 2.0 * shift_px[:, 0] / float(w - 1)
        if h > 1:
            theta[:, 1, 2] = 2.0 * shift_px[:, 1] / float(h - 1)

        grid = F.affine_grid(theta, size = thermal.shape, align_corners = False)
        return F.grid_sample(
            thermal,
            grid,
            mode = "bilinear",
            padding_mode = "zeros",
            align_corners = False,
        )

    def forward(self, x4: torch.Tensor, return_aux: bool = False):
        if x4.dim() != 4:
            raise ValueError(f"Expected x4 to be 4D (B,C,H,W), got {tuple(x4.shape)}")
        if x4.shape[1] != 4:
            raise ValueError(f"Expected x4 to have 4 channels, got {x4.shape[1]}")

        rgb = x4[:, :3, :, :]
        thermal = x4[:, 3:4, :, :]

        pred_shift_px, thermal_conf = self._encode_alignment(x4)
        thermal_aligned = self._warp_thermal(thermal, pred_shift_px)
        thermal_weighted = thermal_aligned * thermal_conf.view(-1, 1, 1, 1)
        x4_fused = torch.cat([rgb, thermal_weighted], dim = 1)

        density = super().forward(x4_fused)
        if not return_aux:
            return density

        aux = {
            "pred_shift_px": pred_shift_px,
            "thermal_conf": thermal_conf.view(-1),
        }
        return density, aux
