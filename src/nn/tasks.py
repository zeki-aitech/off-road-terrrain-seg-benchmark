
import contextlib

import torch
import torch.nn as nn

from ultralytics.models import yolo
from ultralytics.utils import (
    LOGGER, colorstr
)
from ultralytics.utils.ops import make_divisible

from ultralytics.nn.modules import (
    Conv,
    Concat,
)

from src.nn.modules import (
    DeepLabV3PlusResNet50Backbone,
    ResNet50Stem,
    ResNet50Layer,
    ASPPPooling,
    ASPP,
    SeparableConv,
    DeepLabV3PlusSemanticSegment,
)

from src.utils.loss import DeepLabV3PlusSemanticSegmentationLoss
    

def parse_model(d, ch, verbose=True):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Conv,  # Conv2d, ConvTranspose2d, SeparableConv
            SeparableConv,  # SeparableConv2d
            ResNet50Stem,  # ResNet50 Stem
            ResNet50Layer,  # ResNet50 Layer
            ASPP,  # Atrous Spatial Pyramid Pooling
            ASPPPooling,  # ASPP Pooling Layer
            DeepLabV3PlusSemanticSegment,  # DeepLabV3+ Semantic Segmentation Head
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)



class DeepLabV3PlusSemanticSegmentationModel(yolo.model.SegmentationModel):
    """
    DeepLabV3+ model for semantic segmentation.
    This class extends the YOLO SegmentationModel to implement the DeepLabV3+ architecture for semantic segmentation tasks.
    
    """
    
    def __init__(self, cfg="deeplabv3plus_resnet50.yaml", ch=3, nc=None, verbose=True):
        """
        Initializes the DeepLabV3Plus model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
    
    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return DeepLabV3PlusSemanticSegmentationLoss(self)
    