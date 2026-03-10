import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class CSRNetRGBT_AdaptiveFPNLite(nn.Module):
    def __init__(self, load_imagenet: bool = True, feat_ch: int = 64):
        super().__init__()
        vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1 if load_imagenet else None)
        feats = list(vgg.features.children())

        conv1 = feats[0]
        new_conv1 = nn.Conv2d(4, 64, kernel_size = 3, padding = 1, bias = True)
        with torch.no_grad():
            if load_imagenet:
                new_conv1.weight[:, :3, :, :] = conv1.weight
                new_conv1.weight[:, 3:4, :, :] = conv1.weight.mean(dim = 1, keepdim = True)
                if conv1.bias is not None:
                    new_conv1.bias.copy_(conv1.bias)
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode = "fan_out", nonlinearity = "relu")
                nn.init.zeros_(new_conv1.bias)
        feats[0] = new_conv1

        self.frontend = nn.Sequential(*feats[:23])
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

        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for m in [self.lat2, self.lat3, self.lat4, self.latb, self.scale_gate, self.residual_out]:
            nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)
        nn.init.zeros_(self.residual_out.weight)

        self._idx_c2 = 8
        self._idx_c3 = 15
        self._idx_c4 = 22

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

    def _forward_frontend_feats(self, x: torch.Tensor):
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

        c2, c3, c4 = self._forward_frontend_feats(x4)
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
        return F.softplus(base_den + residual)
