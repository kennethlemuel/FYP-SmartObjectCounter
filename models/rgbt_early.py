import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class CSRNetRGBT_Early(nn.Module):
    def __init__(self, load_imagenet = True):
        super().__init__()
        vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1 if load_imagenet else None)
        feats = list(vgg.features.children())
        #replacing the first conv to accept 4 channels
        conv1 = feats[0]
        new_conv1 = nn.Conv2d(4, 64, kernel_size = 3, padding = 1, bias = True)

        with torch.no_grad():
            if load_imagenet:
                #copying RGB weights
                new_conv1.weight[:, :3, :, :] = conv1.weight
                #4th channel = mean of the 3 RGB channels
                new_conv1.weight[:, 3:4, :, :] = conv1.weight.mean(dim = 1, keepdim = True)
                if conv1.bias is not None:
                    new_conv1.bias.copy_(conv1.bias)
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode = 'fan_out', nonlinearity = 'relu')
                nn.init.zeros_(new_conv1.bias)

        feats[0] = new_conv1
        self.frontend = nn.Sequential(*feats[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(512, 256, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, 3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
            nn.Conv2d(128, 64,  3, padding = 2, dilation = 2), nn.ReLU(inplace = True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1, bias = False)

        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x_rgb, x_t):
        """Early fusion: concatenate RGB (3ch) + Thermal (1ch) -> 4ch."""
        if x_rgb.dim() != 4 or x_t.dim() != 4:
            raise ValueError(f"Expected x_rgb and x_t to be 4D tensors, got {tuple(x_rgb.shape)} and {tuple(x_t.shape)}")

        # Thermal can arrive as 1ch or 3ch; normalize to 1ch for 4-channel fusion.
        if x_t.shape[1] == 3:
            x_t = x_t.mean(dim = 1, keepdim = True)
        elif x_t.shape[1] != 1:
            raise ValueError(f"Expected x_t to have 1 or 3 channels, got {x_t.shape[1]}")
        if x_rgb.shape[1] != 3:
            raise ValueError(f"Expected x_rgb to have 3 channels, got {x_rgb.shape[1]}")

        if x_rgb.shape[0] != x_t.shape[0] or x_rgb.shape[2:] != x_t.shape[2:]:
            raise ValueError(f"x_rgb and x_t must share batch/spatial dims, got {tuple(x_rgb.shape)} vs {tuple(x_t.shape)}")

        x4 = torch.cat([x_rgb, x_t], dim = 1)

        x = self.frontend(x4)
        x = self.backend(x)
        x = self.output_layer(x)
        return F.softplus(x)
