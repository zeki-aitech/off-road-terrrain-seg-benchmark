from torch import nn
from torchvision import models as tvmodels
from torchvision.models import ResNet50_Weights

__all__ = "ResNet50Backbone"


class ResNet50Backbone(nn.Module):
    """
    ResNet50 Backbone for Feature Extraction.

    This class loads a pretrained ResNet50 model (IMAGENET1K_V2 weights by default),
    removes the average pooling and fully connected layers, and exposes the convolutional
    feature extractor as a backbone.

    Output:
        For input of shape (B, 3, H, W), output will be (B, 2048, H/32, W/32).
        For example, input (B, 3, 224, 224) -> output (B, 2048, 7, 7).
    """
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        resnet50 = tvmodels.resnet50(weights=weights)
        # Remove avgpool and fc layers to get the backbone
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            Tensor: Feature map of shape (B, 2048, H/32, W/32)
        """
        return self.backbone(x)
    

class AtrousSeparableConv(nn.Module):
    pass