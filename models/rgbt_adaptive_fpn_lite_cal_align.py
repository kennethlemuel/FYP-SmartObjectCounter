import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgbt_adaptive_fpn_lite_cal import CSRNetRGBT_AdaptiveFPNLiteCal


class CSRNetRGBT_AdaptiveFPNLiteCalAlign(CSRNetRGBT_AdaptiveFPNLiteCal):
    """
    Amendment Version 2:
    Adaptive FPN Lite + Calibration with a lightweight learnable thermal
    pre-alignment block. The RGB stream is kept unchanged; only the thermal
    channel is shifted by a small predicted global offset before fusion.
    """

    def __init__(
        self,
        load_imagenet: bool = True,
        feat_ch: int = 64,
        scale_bound: float = 0.1,
        max_shift_px: float = 4.0,
    ):
        super().__init__(load_imagenet=load_imagenet, feat_ch=feat_ch, scale_bound=scale_bound)
        self.max_shift_px = float(max_shift_px)

        self.align_feat = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.align_fc = nn.Linear(16, 2, bias=True)

        for m in self.align_feat.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Identity / zero-shift initialization.
        nn.init.zeros_(self.align_fc.weight)
        nn.init.zeros_(self.align_fc.bias)

    def _predict_shift(self, x4: torch.Tensor) -> torch.Tensor:
        feat = self.align_feat(x4).flatten(1)
        shift = torch.tanh(self.align_fc(feat)) * self.max_shift_px
        return shift

    def _align_thermal(self, x4: torch.Tensor) -> torch.Tensor:
        rgb = x4[:, :3, :, :]
        t = x4[:, 3:4, :, :]

        shift = self._predict_shift(x4)
        dx = shift[:, 0]
        dy = shift[:, 1]

        b, _, h, w = t.shape
        theta = t.new_zeros((b, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0

        # affine_grid expects normalized translations in [-1, 1]
        if w > 1:
            theta[:, 0, 2] = 2.0 * dx / float(w - 1)
        if h > 1:
            theta[:, 1, 2] = 2.0 * dy / float(h - 1)

        grid = F.affine_grid(theta, size=t.shape, align_corners=False)
        t_aligned = F.grid_sample(
            t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return torch.cat([rgb, t_aligned], dim=1)

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        x4_aligned = self._align_thermal(x4)
        return super().forward(x4_aligned)
