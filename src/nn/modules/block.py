import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tvmodels
from torchvision.models import ResNet50_Weights

from ultralytics.nn.modules import conv as ul_conv

__all__ = "DeepLabV3PlusResNet50Backbone", "ASPPPooling", "ASPP"


class DeepLabV3PlusResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone for DeepLabv3+ that outputs both high-level and low-level features.
    - high-level: output of layer4 (2048 channels, 1/16 input resolution)
    - low-level: output of layer1 (256 channels, 1/4 input resolution)
    """
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        resnet50 = tvmodels.resnet50(weights=weights)
        # Stem: initial layers before residual blocks
        self.stem = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )
        self.layer1 = resnet50.layer1  # Low-level features (256 channels)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4  # High-level features (2048 channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W)
        Returns:
            Tuple[Tensor, Tensor]:
                - high_level_features: (B, 2048, H/16, W/16)
                - low_level_features: (B, 256, H/4, W/4)
        """
        x = self.stem(x)
        low_level_features = self.layer1(x)           # (B, 256, H/4, W/4)
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        high_level_features = self.layer4(x)          # (B, 2048, H/16, W/16)
        return high_level_features, low_level_features
    
    
class ASPPPooling(nn.Module):
    """
    ASPP Pooling Layer for Atrous Spatial Pyramid Pooling.
    """
    def __init__(self, c1, c2):
        """
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ul_conv.Conv(c1, c2, k=1, s=1, p=0, g=1, d=1, act=True)

    def forward(self, x):
        """
        Forward pass through the ASPP pooling layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor of shape (B, C', 1, 1)
        """
        size = x.shape[-2:]  # Get input spatial size
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module for DeepLabV3+.

    This module implements the ASPP block with multiple atrous convolutions
    at different rates to capture multi-scale context.
    """
    
    def __init__(self, c1, c2, atrous_rates=(6, 12, 18)):
        
        super().__init__()
        modules = []
        # 1x1 convolution
        modules.append(ul_conv.Conv(c1, c2, k=1, s=1, p=0, g=1, d=1, act=True))
        
        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(ul_conv.Conv(c1, c2, k=3, s=1, p=rate, g=1, d=rate, act=True))
        
        # ASPP pooling
        modules.append(ASPPPooling(c1, c2))
        
        self.convs = nn.ModuleList(modules)
        
        self.projector = ul_conv.Conv(len(modules) * c2, c2, k=1, s=1, p=0, g=1, d=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through the ASPP module.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor of shape (B, C', H, W)
        """
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.projector(x)
        return x

        
        
        
        
    
    
    