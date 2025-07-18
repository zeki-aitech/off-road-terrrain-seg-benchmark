from typing import Optional

from torch import nn

from ultralytics.nn.modules import (
    Conv,
)


__all__ = "DeepLabV3PlusSemanticSegment"



class DeepLabV3PlusSemanticSegment(nn.Module):
    """
    DeepLabV3+ Semantic Segmentation Head
    """
    def __init__(self, c1, nc):
        """
        Args:
            c1 (int): Input channels.
            nc (int): Number of classes for segmentation.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, nc, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)  # Along the class dimension

    def forward(self, x):
        logits = self.conv(x)
        
        if self.training:
            # Return raw logits during training (for use with cross-entropy loss)
            return logits
        
        # Return class probabilities during evaluation
        probs = self.softmax(logits)
        return probs