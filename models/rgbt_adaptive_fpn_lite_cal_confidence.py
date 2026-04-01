import torch
import torch.nn as nn

from models.rgbt_adaptive_fpn_lite_cal import CSRNetRGBT_AdaptiveFPNLiteCal


class CSRNetRGBT_AdaptiveFPNLiteCalConfidence(CSRNetRGBT_AdaptiveFPNLiteCal):
    """
    Amendment Version 4:
    Confidence-aware Adaptive FPN Lite + Calibration for asynchronous RGB-T data.

    Rather than forcing strict alignment, the model predicts a local thermal
    confidence map from RGB intensity, thermal intensity, and their absolute
    disagreement. The thermal channel is softly attenuated where cross-modal
    agreement is weak before entering the shared backbone.
    """

    def __init__(
        self,
        load_imagenet: bool = True,
        feat_ch: int = 64,
        scale_bound: float = 0.1,
        thermal_conf_floor: float = 0.35,
    ):
        super().__init__(load_imagenet = load_imagenet, feat_ch = feat_ch, scale_bound = scale_bound)
        self.thermal_conf_floor = float(thermal_conf_floor)

        self.conf_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 8, 3, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(8, 1, 1, bias = True),
        )

        for m in self.conf_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last = self.conf_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 2.0)

    def _confidence_map(self, x4: torch.Tensor) -> torch.Tensor:
        rgb = x4[:, :3, :, :]
        thermal = x4[:, 3:4, :, :]
        rgb_gray = rgb.mean(dim = 1, keepdim = True)
        disagree = (rgb_gray - thermal).abs()
        conf_in = torch.cat([rgb_gray, thermal, disagree], dim = 1)
        conf = torch.sigmoid(self.conf_head(conf_in))
        conf = self.thermal_conf_floor + (1.0 - self.thermal_conf_floor) * conf
        return conf

    def forward(self, x4: torch.Tensor, return_aux: bool = False):
        if x4.dim() != 4:
            raise ValueError(f"Expected x4 to be 4D (B,C,H,W), got {tuple(x4.shape)}")
        if x4.shape[1] != 4:
            raise ValueError(f"Expected x4 to have 4 channels, got {x4.shape[1]}")

        rgb = x4[:, :3, :, :]
        thermal = x4[:, 3:4, :, :]
        thermal_conf = self._confidence_map(x4)
        x4_fused = torch.cat([rgb, thermal * thermal_conf], dim = 1)

        density = super().forward(x4_fused)
        if not return_aux:
            return density

        aux = {
            "thermal_conf_mean": thermal_conf.mean(dim = (1, 2, 3)),
        }
        return density, aux
