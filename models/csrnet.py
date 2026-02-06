import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class CSRNet(nn.Module):
    def __init__(self, load_imagenet: bool = True):
        super().__init__()
        vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1 if load_imagenet else None)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[:23])
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
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.softplus(x)
        return x

        