import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class CSRNetRGBT_Base(nn.Module):
    """
    Base RGBT model:
    - Single-stream CSRNet with a 4-channel input (RGB + Thermal).
    - No explicit fusion layers beyond the 4-channel input.
    """
    def __init__(self, load_imagenet: bool = True):
        super().__init__()
        vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1 if load_imagenet else None)
        feats = list(vgg.features.children())

        conv1 = feats[0]
        new_conv1 = nn.Conv2d(4, 64, kernel_size = 3, padding = 1, bias = True)

        with torch.no_grad():
            if load_imagenet:
                # Copy RGB weights; initialize thermal channel as mean of RGB filters.
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

        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        if x4.dim() != 4:
            raise ValueError(f"Expected x4 to be 4D (B,C,H,W), got {tuple(x4.shape)}")
        if x4.shape[1] != 4:
            raise ValueError(f"Expected x4 to have 4 channels, got {x4.shape[1]}")

        x = self.frontend(x4)
        x = self.backend(x)
        x = self.output_layer(x)
        return F.softplus(x)
