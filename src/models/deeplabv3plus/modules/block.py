import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class ResNet50BackBoneBlock(nn.Module):
    def __init__(self, c1, c2, weights=None):
        """
        Args:
            c1: Number of input channels 
            c2: Number of output channels
            weights: Pre-trained weights for ResNet50.
        """
        super().__init__()
        resnet50_block = resnet50(weights=weights)
        
        self.block = IntermediateLayerGetter(
            resnet50_block,
            return_layers={
                "layer1": "low_level",  # Output from the first layer of ResNet50
                "layer4": "out",  # Output from the last layer of ResNet50
            }
        )
        
    def forward(self, x):
        """
        Forward pass through the ResNet50 block.
        
        Args:
            x: Input tensor.
        
        Returns:
            dict: Dictionary containing outputs from specified layers.
        """
        out= self.block(x)
        return out


# test
# if __name__ == "__main__":
#     model = ResNet50BackBoneBlock(c1=3, c2=2048, weights=ResNet50_Weights.IMAGENET1K_V1)
#     x = torch.randn(1, 3, 224, 224)  # Example input tensor
#     output = model(x)
#     print(output.keys())  # Should print the keys of the output dictionary
#     print(output['low_level'].shape)  # Output from the first layer
#     print(output['out'].shape)  # Output from the last layer
#     print(output)



class AtrousSeparableConvolution(nn.Module):
    """ 
    Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)