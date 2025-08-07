from typing import Optional

import torch
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
        
        # Initialize weights for better stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Add stability check for input
        if torch.isnan(x).any():
            # Replace NaN inputs with zeros to prevent propagation
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Clamp extreme values to prevent overflow in mixed precision
        x = torch.clamp(x, min=-50.0, max=50.0)
        
        logits = self.conv(x)
        
        # Additional stability check for outputs
        if torch.isnan(logits).any():
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        
        # Clamp logits to prevent fp16 overflow
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        if self.training:
            # Return raw logits during training (for use with cross-entropy loss)
            return logits
        
        # Return class probabilities during evaluation
        probs = self.softmax(logits)
        return probs