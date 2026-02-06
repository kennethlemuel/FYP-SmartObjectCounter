import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


class _MultiScaleGateNet(nn.Module):
    """
    Multi-scale spatial gate.

    Inputs: pred_rgb, pred_t, each [B, 1, H, W]
    Output: gate in [0, 1], shape [B, 1, H, W]

    Uses 6-channel gating input to match older checkpoints:
      [pred_rgb, pred_t, |diff|, mean, max, min]
    """

    def __init__(self, hidden: int = 32, scales = (1, 2, 4)):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        if len(self.scales) == 0:
            raise ValueError("scales must be non-empty")
        if any(s <= 0 for s in self.scales):
            raise ValueError(f"All scales must be positive, got {self.scales}")

        self.conv1 = nn.Conv2d(6, hidden, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size = 1, padding = 0)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x), inplace = True)
        return self.conv2(h)

    def forward(self, pred_rgb: torch.Tensor, pred_t: torch.Tensor, return_aux: bool = False):
        diff = (pred_rgb - pred_t).abs()
        mean = 0.5 * (pred_rgb + pred_t)
        mx = torch.maximum(pred_rgb, pred_t)
        mn = torch.minimum(pred_rgb, pred_t)

        x = torch.cat([pred_rgb, pred_t, diff, mean, mx, mn], dim = 1)
        _, _, H, W = x.shape

        logits_scales = []
        for s in self.scales:
            if s == 1:
                x_s = x
            else:
                x_s = F.avg_pool2d(x, kernel_size = s, stride = s)

            logit_s = self._logits(x_s)
            if logit_s.shape[-2:] != (H, W):
                logit_s = F.interpolate(logit_s, size = (H, W), mode = "bilinear", align_corners = False)
            logits_scales.append(logit_s)

        logits = torch.stack(logits_scales, dim = 0).mean(dim = 0)
        gate = torch.sigmoid(logits)

        if not return_aux:
            return gate

        aux = {
            "gate": gate,
            "gate_logits": logits,
            "gate_logits_scales": logits_scales,
        }
        return gate, aux


class CSRNetRGBT_AdaptiveLate(nn.Module):
    """Adaptive late fusion: two CSRNet experts + (optional) per-modality calibration + multi-scale gate."""

    def __init__(
        self,
        load_imagenet: bool | None = None,
        load_weights: bool | None = None,  # legacy alias
        gate_hidden: int = 32,
        gate_scales = (1, 2, 4),
        output_size = None,
        count_preserve_resize: bool = True,
        use_calibration: bool = True,
    ):
        super().__init__()

        if load_imagenet is None:
            load_imagenet = True if load_weights is None else bool(load_weights)
        else:
            if load_weights is not None and bool(load_imagenet) != bool(load_weights):
                raise ValueError("Conflicting values: load_imagenet and load_weights differ.")

        self.rgb_net = CSRNet(load_imagenet = bool(load_imagenet))
        self.t_net = CSRNet(load_imagenet = bool(load_imagenet))

        self.use_calibration = bool(use_calibration)
        if self.use_calibration:
            self.rgb_cal = nn.Conv2d(1, 1, 1, bias = True)
            self.t_cal = nn.Conv2d(1, 1, 1, bias = True)
            with torch.no_grad():
                self.rgb_cal.weight.fill_(1.0)
                self.rgb_cal.bias.zero_()
                self.t_cal.weight.fill_(1.0)
                self.t_cal.bias.zero_()

        self.gate = _MultiScaleGateNet(hidden = gate_hidden, scales = gate_scales)

        self.output_size = output_size
        self.count_preserve_resize = bool(count_preserve_resize)

    @property
    def gate_net(self):
        # Expose gate_net for code that expects it, without registering a second module in state_dict.
        return self.gate

    @staticmethod
    def _resize_density_sum_preserving(density: torch.Tensor, size) -> torch.Tensor:
        if density.shape[-2:] == tuple(size):
            return density
        in_sum = density.sum(dim = (2, 3), keepdim = True)
        out = F.interpolate(density, size = size, mode = "bilinear", align_corners = False)
        out_sum = out.sum(dim = (2, 3), keepdim = True).clamp_min(1e-6)
        return out * (in_sum / out_sum)

    def _maybe_resize(self, pred: torch.Tensor) -> torch.Tensor:
        if self.output_size is None:
            return pred
        if self.count_preserve_resize:
            return self._resize_density_sum_preserving(pred, self.output_size)
        return F.interpolate(pred, size = self.output_size, mode = "bilinear", align_corners = False)

    def _match_pair(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a.shape[-2:] == b.shape[-2:]:
            return a, b
        b2 = self._resize_density_sum_preserving(b, a.shape[-2:])
        return a, b2

    def _apply_cal(self, pred_rgb: torch.Tensor, pred_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_calibration:
            return pred_rgb, pred_t
        pred_rgb = self.rgb_cal(pred_rgb)
        pred_t = self.t_cal(pred_t)
        return pred_rgb, pred_t

    def forward(self, x_rgb: torch.Tensor, x_t3: torch.Tensor) -> torch.Tensor:
        pred_rgb = self._maybe_resize(self.rgb_net(x_rgb))
        pred_t = self._maybe_resize(self.t_net(x_t3))
        pred_rgb, pred_t = self._match_pair(pred_rgb, pred_t)

        pred_rgb, pred_t = self._apply_cal(pred_rgb, pred_t)

        gate = self.gate(pred_rgb, pred_t)
        pred = gate * pred_rgb + (1.0 - gate) * pred_t
        return F.softplus(pred)

    def forward_with_aux(self, x_rgb: torch.Tensor, x_t3: torch.Tensor):
        pred_rgb = self._maybe_resize(self.rgb_net(x_rgb))
        pred_t = self._maybe_resize(self.t_net(x_t3))
        pred_rgb, pred_t = self._match_pair(pred_rgb, pred_t)

        pred_rgb, pred_t = self._apply_cal(pred_rgb, pred_t)

        gate, gate_aux = self.gate(pred_rgb, pred_t, return_aux = True)
        pred = F.softplus(gate * pred_rgb + (1.0 - gate) * pred_t)

        aux = {
            "pred_rgb": pred_rgb,
            "pred_t": pred_t,
        }
        aux.update(gate_aux)
        return pred, aux
