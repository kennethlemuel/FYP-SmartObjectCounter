import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import CSRNet


class _MultiScaleGateNet(nn.Module):
    """
    Multi-scale spatial gate with minimal extra compute.

    Given two modality predictions pred_rgb and pred_t (both [B,1,H,W]),
    compute a per-location gate g in [0,1] to mix:
        pred = g * pred_rgb + (1 - g) * pred_t

    We build a 4-channel summary map:
        x = [pred_rgb, pred_t, |pred_rgb - pred_t|, 0.5*(pred_rgb + pred_t)]

    Then compute gate logits at multiple scales via average pooling, upsample
    logits back to (H,W), average logits, then apply sigmoid.

    Last conv is zero-initialized so the initial gate is ~0.5 everywhere.
    """

    def __init__(self, hidden: int = 32, scales = (1, 2, 4)):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        if len(self.scales) == 0:
            raise ValueError("scales must be non-empty")
        if any(s <= 0 for s in self.scales):
            raise ValueError(f"All scales must be positive, got {self.scales}")

        self.conv1 = nn.Conv2d(4, hidden, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size = 1, padding = 0)

        # Start from an unbiased 0.5 gate everywhere.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x), inplace = True)
        return self.conv2(h)

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_t: torch.Tensor,
        return_aux: bool = False,
    ):
        x = torch.cat(
            [pred_rgb, pred_t, (pred_rgb - pred_t).abs(), 0.5 * (pred_rgb + pred_t)],
            dim = 1,
        )
        _, _, H, W = x.shape

        logits_scales = []
        for s in self.scales:
            if s == 1:
                x_s = x
            else:
                x_s = F.avg_pool2d(x, kernel_size = s, stride = s)

            logit_s = self._logits(x_s)

            if logit_s.shape[-2:] != (H, W):
                logit_s = F.interpolate(
                    logit_s,
                    size = (H, W),
                    mode = "bilinear",
                    align_corners = False,
                )

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
    """
    Adaptive Late Fusion: two CSRNet experts + multi-scale spatial gating.
    """

    def __init__(
        self,
        load_imagenet: bool = True,
        load_weights: bool = None,          # backward-compatible alias
        gate_hidden: int = 32,
        gate_scales = (1, 2, 4),
        output_size = None,
        count_preserve_resize: bool = True,
        **kwargs,                           # swallow any extra unexpected args safely
    ):
        super().__init__()

        # Backward compatibility:
        # - old code might pass load_weights
        # - new code passes load_imagenet
        if load_weights is not None:
            load_imagenet = bool(load_weights)

        self.rgb_net = CSRNet(load_imagenet = bool(load_imagenet))
        self.t_net = CSRNet(load_imagenet = bool(load_imagenet))

        self.gate_net = _MultiScaleGateNet(hidden = gate_hidden, scales = gate_scales)
        # Backward-compatible alias for older training code
        self.gate = self.gate_net

        # If set, force both expert outputs to this (H, W) before gating.
        self.output_size = output_size
        self.count_preserve_resize = bool(count_preserve_resize)

    @staticmethod
    def _resize_density_sum_preserving(density: torch.Tensor, size) -> torch.Tensor:
        """
        Resize density while preserving integral (sum), so the predicted count
        does not change due to interpolation.
        """
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

    def forward(self, x_rgb: torch.Tensor, x_t3: torch.Tensor) -> torch.Tensor:
        pred_rgb = self._maybe_resize(self.rgb_net(x_rgb))
        pred_t = self._maybe_resize(self.t_net(x_t3))

        gate = self.gate_net(pred_rgb, pred_t)
        pred = gate * pred_rgb + (1.0 - gate) * pred_t
        return pred

    def forward_with_aux(self, x_rgb: torch.Tensor, x_t3: torch.Tensor):
        """
        Return (pred, aux) so train/eval can log gate behavior.
        """
        pred_rgb = self._maybe_resize(self.rgb_net(x_rgb))
        pred_t = self._maybe_resize(self.t_net(x_t3))

        gate, gate_aux = self.gate_net(pred_rgb, pred_t, return_aux = True)
        pred = gate * pred_rgb + (1.0 - gate) * pred_t

        aux = {
            "pred_rgb": pred_rgb,
            "pred_t": pred_t,
        }
        aux.update(gate_aux)
        return pred, aux
