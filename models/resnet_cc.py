import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetCount(nn.Module):
    """
    ResNet-50 backbone with a lightweight density head.
    Outputs density at stride 8 (c3 resolution), matching out_stride=8 targets.
    """
    def __init__(self, in_ch: int = 3, load_imagenet: bool = True, head_channels: int = 256):
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

        # Backbone layers (names match ResNetBackbone used in FPN)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        # Density head on c3 (stride 8)
        self.head = nn.Sequential(
            nn.Conv2d(512, head_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(head_channels, head_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(head_channels, 1, 1, bias = False),
        )

        for mod in self.head.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode = "fan_out", nonlinearity = "relu")
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        _c5 = self.layer4(c4)
        den = self.head(c3)
        return F.softplus(den)
