# src/utils/loss.py

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER, colorstr


class DeepLabV3PlusSemanticSegmentationLoss:
    """
    DeepLabV3+ loss function for semantic segmentation.
    This class implements the loss function used in the DeepLabV3+ model.
    """
    
    def __init__(self, model):
        """
        Initializes the DeepLabV3PlusLoss.
        """
        super(DeepLabV3PlusSemanticSegmentationLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        
        self.device = device
        
    def __call__(
        self,
        preds: Any, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for the DeepLabV3+ model.

        Args:
            preds (Any): Model predictions.
            batch (Dict[str, torch.Tensor]): Batch of data containing input images and target masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss value and additional loss items.
        """
        
        # as we'd known that:
        # preds: [B, C=(number of class), H, W]
        # batch["mask"]: [B, C=1, H, W]
        # need to calcualte the loss between preds and batch["mask"]
        # loss --> cross entropy loss
        
        # assume preds and batch["mask"] are already on the same device
        
        # resize the mask to match the prediction shape
        
        masks = batch["masks"]
        
        # Move mask to model device
        masks = masks.to(self.device)
        
        # Resize and ensure shape
        masks = F.interpolate(masks.float().unsqueeze(1), size=preds.shape[2:], mode='nearest').squeeze(1)
        masks = masks.long()
        
        # Sanity checks (optional)
        if preds.shape[0] != masks.shape[0] or preds.shape[2:] != masks.shape[1:]:
            raise ValueError("Shape mismatch between preds and mask.")
        
        loss = F.cross_entropy(preds, masks, reduction='mean')
        if torch.isnan(loss):
            raise ValueError("Loss is NaN; check your data and model.")
        return loss, loss.detach()
        
        
        
    
        
        
        
        