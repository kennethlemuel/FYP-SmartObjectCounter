import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


class CSRNetRGBT_AdaptiveFPNLiteCalDualStem(nn.Module):
    """
    Amendment Version 5:
    Adaptive FPN Lite + Calibration with shallow dual stems before fusion.

    RGB and thermal are processed separately through a lightweight first conv
    block, then fused and passed into the original shared lightweight adaptive
    FPN + calibration path. This keeps the model lightweight while relaxing the
    strongest early-fusion assumption of raw 4-channel stacking.
    """

    def __init__(self, load_imagenet: bool = True, feat_ch: int = 64, scale_bound: float = 0.1):
        super().__init__()
        vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1 if load_imagenet else None)
        feats = list(vgg.features.children())

        conv1 = feats[0]  # 3 -> 64
        conv2 = feats[2]  # 64 -> 64

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace = True),
        )
        self.t_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace = True),
        )
        self.stem_fuse = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1, bias = True),
            nn.ReLU(inplace = True),
        )

        with torch.no_grad():
            rgb_conv1 = self.rgb_stem[0]
            t_conv1 = self.t_stem[0]
            if load_imagenet:
                rgb_conv1.weight.copy_(conv1.weight[:32])
                if conv1.bias is not None:
                    rgb_conv1.bias.copy_(conv1.bias[:32])
                t_conv1.weight.copy_(conv1.weight[:32].mean(dim = 1, keepdim = True))
                if conv1.bias is not None:
                    t_conv1.bias.copy_(conv1.bias[:32])
            else:
                nn.init.kaiming_normal_(rgb_conv1.weight, mode = "fan_out", nonlinearity = "relu")
                nn.init.zeros_(rgb_conv1.bias)
                nn.init.kaiming_normal_(t_conv1.weight, mode = "fan_out", nonlinearity = "relu")
                nn.init.zeros_(t_conv1.bias)

        for stem in (self.rgb_stem, self.t_stem):
            second = stem[2]
            nn.init.kaiming_normal_(second.weight, mode = "fan_out", nonlinearity = "relu")
            nn.init.zeros_(second.bias)

        fuse_conv = self.stem_fuse[0]
        if load_imagenet:
            with torch.no_grad():
                fuse_conv.weight.zero_()
                # Approximate the original conv2 by averaging its response over the split stems.
                fuse_conv.weight[:, :32] = conv2.weight[:, :32]
                fuse_conv.weight[:, 32:] = conv2.weight[:, 32:]
                if conv2.bias is not None:
                    fuse_conv.bias.copy_(conv2.bias)
                else:
                    nn.init.zeros_(fuse_conv.bias)
        else:
            nn.init.kaiming_normal_(fuse_conv.weight, mode = "fan_out", nonlinearity = "relu")
            nn.init.zeros_(fuse_conv.bias)

        # Continue from the first pooling layer onward.
        self.frontend = nn.Sequential(*feats[4:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(512, 256, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1, bias = False)

        self.lat2 = nn.Conv2d(128, feat_ch, 1, bias = False)
        self.lat3 = nn.Conv2d(256, feat_ch, 1, bias = False)
        self.lat4 = nn.Conv2d(512, feat_ch, 1, bias = False)
        self.latb = nn.Conv2d(64, feat_ch, 1, bias = False)
        self.scale_gate = nn.Conv2d(feat_ch * 4, 4, 1, bias = True)
        self.refine = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding = 1, bias = False),
            nn.ReLU(inplace = True),
        )
        self.residual_out = nn.Conv2d(feat_ch, 1, 1, bias = False)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.count_head = nn.Sequential(
            nn.Linear(64, 16, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(16, 1, bias = True),
        )
        self.scale_bound = float(scale_bound)

        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for m in [self.lat2, self.lat3, self.lat4, self.latb, self.scale_gate, self.residual_out]:
            nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)

        for m in self.count_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)
        nn.init.zeros_(self.residual_out.weight)

        self._idx_c2 = 4
        self._idx_c3 = 11
        self._idx_c4 = 18

    @staticmethod
    def _reduce_to(x: torch.Tensor, target_hw):
        th, tw = int(target_hw[0]), int(target_hw[1])
        h, w = x.shape[-2:]
        if h == th and w == tw:
            return x
        if h % th != 0 or w % tw != 0:
            raise ValueError(f"Non-integer resize from {(h, w)} to {(th, tw)} is not supported.")
        sh = h // th
        sw = w // tw
        b, c = x.shape[:2]
        return x.reshape(b, c, th, sh, tw, sw).mean(dim = (3, 5))

    def _forward_frontend_feats(self, x_rgb: torch.Tensor, x_t: torch.Tensor):
        rgb = self.rgb_stem(x_rgb)
        thermal = self.t_stem(x_t)
        x = self.stem_fuse(torch.cat([rgb, thermal], dim = 1))

        c2 = None
        c3 = None
        c4 = None
        for i, layer in enumerate(self.frontend):
            x = layer(x)
            if i == self._idx_c2:
                c2 = x
            elif i == self._idx_c3:
                c3 = x
            elif i == self._idx_c4:
                c4 = x
        return c2, c3, c4

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        if x4.dim() != 4:
            raise ValueError(f"Expected x4 to be 4D (B,C,H,W), got {tuple(x4.shape)}")
        if x4.shape[1] != 4:
            raise ValueError(f"Expected x4 to have 4 channels, got {x4.shape[1]}")

        x_rgb = x4[:, :3, :, :]
        x_t = x4[:, 3:4, :, :]

        c2, c3, c4 = self._forward_frontend_feats(x_rgb, x_t)
        b = self.backend(c4)
        base_den = self.output_layer(b)

        target_hw = b.shape[-2:]
        f2 = self.lat2(self._reduce_to(c2, target_hw))
        f3 = self.lat3(self._reduce_to(c3, target_hw))
        f4 = self.lat4(c4)
        fb = self.latb(b)

        stacked = torch.cat([f2, f3, f4, fb], dim = 1)
        gate = torch.softmax(self.scale_gate(stacked), dim = 1)
        fused = (
            gate[:, 0:1] * f2 +
            gate[:, 1:2] * f3 +
            gate[:, 2:3] * f4 +
            gate[:, 3:4] * fb
        )

        refine = self.refine(torch.cat([fb, fused], dim = 1))
        residual = self.residual_out(refine)

        pooled = b.mean(dim = (-2, -1))
        count_scale = 1.0 + self.scale_bound * torch.tanh(self.count_head(pooled))
        density = F.softplus(base_den + self.residual_scale.view(1, 1, 1, 1) * residual)
        return density * count_scale.view(-1, 1, 1, 1)
