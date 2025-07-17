from typing import Optional

from torch import nn

from ultralytics.nn.modules import (
    Conv,
)


__all__ = "DeepLabV3PlusSemanticSegment"


class DeepLabV3PlusSemanticSegment(nn.Module):
    
    def __init__(self, c1, c2, nc = 80):
        super().__init__()
        
        self.conv = Conv(c1, c2, k=1, s=1, p=1, act=True)
        
    
    def forward(self, x):
        
        x = self.conv(x)
        
        return x