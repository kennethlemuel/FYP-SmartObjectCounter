import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    def __init__(self, in_ch: int = 3, load_imagenet: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if load_imagenet else None
        m = resnet50(weights = weights)
        if in_ch != 3:
            conv1 = m.conv1
            new_conv1 = nn.Conv2d(
                in_ch,
                conv1.out_channels,
                kernel_size = conv1.kernel_size,
                stride = conv1.stride,
                padding = conv1.padding,
                bias = False,
            )
            with torch.no_grad():
                if load_imagenet:
                    if in_ch == 1:
                        new_conv1.weight.copy_(conv1.weight.mean(dim = 1, keepdim = True))
                    else:
                        new_conv1.weight[:, :3].copy_(conv1.weight)
                        if in_ch > 3:
                            extra = conv1.weight.mean(dim = 1, keepdim = True).repeat(1, in_ch - 3, 1, 1)
                            new_conv1.weight[:, 3:in_ch].copy_(extra)
                else:
                    nn.init.kaiming_normal_(new_conv1.weight, mode = "fan_out", nonlinearity = "relu")
            m.conv1 = new_conv1

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class AdaptiveFPNRGBT(nn.Module):
    def __init__(self, load_imagenet: bool = True, fpn_channels: int = 256):
        super().__init__()
        self.rgb = ResNetBackbone(in_ch = 3, load_imagenet = load_imagenet)
        self.t = ResNetBackbone(in_ch = 3, load_imagenet = load_imagenet)

        self.g2 = nn.Conv2d(256 + 256, 1, 3, padding = 1)
        self.g3 = nn.Conv2d(512 + 512, 1, 3, padding = 1)
        self.g4 = nn.Conv2d(1024 + 1024, 1, 3, padding = 1)
        self.g5 = nn.Conv2d(2048 + 2048, 1, 3, padding = 1)

        self.lat2 = nn.Conv2d(256, fpn_channels, 1)
        self.lat3 = nn.Conv2d(512, fpn_channels, 1)
        self.lat4 = nn.Conv2d(1024, fpn_channels, 1)
        self.lat5 = nn.Conv2d(2048, fpn_channels, 1)

        head_channels = max(64, fpn_channels // 2)
        self.scale_logits = nn.Parameter(torch.zeros(4))
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(fpn_channels, head_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
        )
        self.den = nn.Conv2d(head_channels, 1, 1, bias = False)

        for m in [self.g2, self.g3, self.g4, self.g5, self.lat2, self.lat3, self.lat4, self.lat5, self.den]:
            nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _fuse(self, r: torch.Tensor, t: torch.Tensor, gate: nn.Module) -> torch.Tensor:
        g = torch.sigmoid(gate(torch.cat([r, t], dim = 1)))
        return g * r + (1.0 - g) * t

    def forward(self, x_rgb: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 4 or x_rgb.dim() != 4:
            raise ValueError("Inputs must be 4D tensors")
        if x_rgb.shape[1] != 3:
            raise ValueError("x_rgb must have 3 channels")
        if x_t.shape[1] == 1:
            x_t = x_t.repeat(1, 3, 1, 1)
        elif x_t.shape[1] != 3:
            raise ValueError("x_t must have 1 or 3 channels")

        r2, r3, r4, r5 = self.rgb(x_rgb)
        t2, t3, t4, t5 = self.t(x_t)

        f2 = self._fuse(r2, t2, self.g2)
        f3 = self._fuse(r3, t3, self.g3)
        f4 = self._fuse(r4, t4, self.g4)
        f5 = self._fuse(r5, t5, self.g5)

        p5 = self.lat5(f5)
        p4 = self.lat4(f4) + F.interpolate(p5, size = f4.shape[-2:], mode = "bilinear", align_corners = False)
        p3 = self.lat3(f3) + F.interpolate(p4, size = f3.shape[-2:], mode = "bilinear", align_corners = False)
        p2 = self.lat2(f2) + F.interpolate(p3, size = f2.shape[-2:], mode = "bilinear", align_corners = False)

        # Use the full pyramid at stride 4 instead of discarding p3/p4/p5.
        p3_up = F.interpolate(p3, size = p2.shape[-2:], mode = "bilinear", align_corners = False)
        p4_up = F.interpolate(p4, size = p2.shape[-2:], mode = "bilinear", align_corners = False)
        p5_up = F.interpolate(p5, size = p2.shape[-2:], mode = "bilinear", align_corners = False)

        level_w = torch.softmax(self.scale_logits, dim = 0)
        x = torch.cat(
            [
                level_w[0] * p2,
                level_w[1] * p3_up,
                level_w[2] * p4_up,
                level_w[3] * p5_up,
            ],
            dim = 1,
        )
        x = self.head(x)
        x = self.den(x)
        return F.softplus(x)
