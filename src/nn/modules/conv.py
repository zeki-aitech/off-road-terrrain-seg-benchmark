from torch import nn

from ultralytics.nn.modules import conv as ul_conv

__all__ = "SeparableConv"


class SeparableConv(nn.Module):
    """
    Separable Convolution Block with Depthwise and Pointwise Convolutions.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        """
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size for depthwise convolution.
            s (int): Stride for depthwise convolution.
            p (int, optional): Padding for depthwise convolution. Defaults to None.
            d (int): Dilation for depthwise convolution.
            act (bool): Whether to apply activation function after convolutions.
        """
        super().__init__()
        # Depthwise: DWConv handles depthwise convolution with dilation and activation
        self.dw = ul_conv.DWConv(c1, c1, k=k, s=s, d=d, act=act)
        # Pointwise: Conv handles 1x1 convolution for channel mixing
        self.pw = ul_conv.Conv(c1, c2, k=1, s=1, p=0, g=1, d=1, act=act)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x
    
        