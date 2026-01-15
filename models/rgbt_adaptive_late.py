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

    Uses a 4-channel summary map:
        x = [pred_rgb, pred_t, |pred_rgb - pred_t|, 0.5*(pred_rgb + pred_t)]
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


def _strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def _extract_state_dict(ckpt_obj) -> dict:
    """
    Accepts:
      - raw state_dict
      - checkpoint dict with common keys
    """
    if isinstance(ckpt_obj, dict):
        # common patterns
        for key in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                ckpt_obj = ckpt_obj[key]
                break

    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint does not contain a usable state_dict dict.")

    ckpt_obj = _strip_module_prefix(ckpt_obj)
    return ckpt_obj


def _filter_by_prefix(sd: dict, prefix: str) -> dict:
    """
    If sd contains keys like 'rgb_net.xxx', return stripped keys 'xxx'.
    Otherwise return empty dict.
    """
    out = {}
    p = prefix
    for k, v in sd.items():
        if k.startswith(p):
            out[k[len(p):]] = v
    return out


class CSRNetRGBT_AdaptiveLate(nn.Module):
    """
    Adaptive Late Fusion: two CSRNet experts + multi-scale spatial gating.

    Compatibility goals:
      - train_rgbt.py may call CSRNetRGBT_AdaptiveLate(load_imagenet = True)
      - train_rgbt.py may access model.gate.parameters()
    """

    def __init__(
        self,
        load_imagenet: bool | None = None,     # alias expected by your train_rgbt.py
        load_weights: bool | None = None,      # original name
        gate_hidden: int = 32,
        gate_scales = (1, 2, 4),
        output_size = None,
        count_preserve_resize: bool = True,
    ):
        super().__init__()

        # Resolve aliasing cleanly
        if load_weights is None:
            load_weights = True if load_imagenet is None else bool(load_imagenet)
        else:
            if load_imagenet is not None and bool(load_imagenet) != bool(load_weights):
                raise ValueError("Conflicting values: load_imagenet and load_weights differ.")

        self.rgb_net = CSRNet(load_weights = load_weights)
        self.t_net = CSRNet(load_weights = load_weights)

        # IMPORTANT: expose attribute name expected by your train script
        self.gate = _MultiScaleGateNet(hidden = gate_hidden, scales = gate_scales)
        # keep an alias too (nice for readability)
        self.gate_net = self.gate

        self.output_size = output_size
        self.count_preserve_resize = bool(count_preserve_resize)

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

    @torch.no_grad()
    def load_pretrained_experts(
        self,
        rgb_ckpt_path: str | None = None,
        t_ckpt_path: str | None = None,
        map_location: str = "cpu",
        strict: bool = False,
    ):
        """
        Warm-start experts from your already-trained baselines.

        Supports both:
          (A) CSRNet-only checkpoints (keys like 'frontend.0.weight', ...)
          (B) full-model checkpoints with prefixes like 'rgb_net.frontend.0.weight'
        """
        if rgb_ckpt_path:
            ckpt = torch.load(rgb_ckpt_path, map_location = map_location)
            sd = _extract_state_dict(ckpt)

            # If it looks like a full model, strip 'rgb_net.'
            sd_rgb = _filter_by_prefix(sd, "rgb_net.")
            if len(sd_rgb) > 0:
                self.rgb_net.load_state_dict(sd_rgb, strict = strict)
            else:
                self.rgb_net.load_state_dict(sd, strict = strict)

        if t_ckpt_path:
            ckpt = torch.load(t_ckpt_path, map_location = map_location)
            sd = _extract_state_dict(ckpt)

            sd_t = _filter_by_prefix(sd, "t_net.")
            if len(sd_t) > 0:
                self.t_net.load_state_dict(sd_t, strict = strict)
            else:
                self.t_net.load_state_dict(sd, strict = strict)

    def forward(self, x_rgb: torch.Tensor, x_t3: torch.Tensor) -> torch.Tensor:
        pred_rgb = self._maybe_resize(self.rgb_net(x_rgb))
        pred_t = self._maybe_resize(self.t_net(x_t3))
        gate = self.gate(pred_rgb, pred_t)
        pred = gate * pred_rgb + (1.0 - gate) * pred_t
        return pred

    def forward_with_aux(self, x_rgb: torch.Tensor, x_t3: torch.Tensor):
        pred_rgb = self._maybe_resize(self.rgb_net(x_rgb))
        pred_t = self._maybe_resize(self.t_net(x_t3))

        gate, gate_aux = self.gate(pred_rgb, pred_t, return_aux = True)
        pred = gate * pred_rgb + (1.0 - gate) * pred_t

        aux = {
            "pred_rgb": pred_rgb,
            "pred_t": pred_t,
        }
        aux.update(gate_aux)
        return pred, aux
