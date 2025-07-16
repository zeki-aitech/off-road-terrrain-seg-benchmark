# src/utils/loss.py

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER, colorstr


class DeepLabV3PlusLoss:
    """
    DeepLabV3+ loss function for semantic segmentation.
    This class implements the loss function used in the DeepLabV3+ model.
    """
    
    def __init__(self, model):
        """
        Initializes the DeepLabV3PlusLoss.
        """
        super(DeepLabV3PlusLoss, self).__init__()
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
        
        # img = batch["img"]
        loss = torch.zeros(1, device=self.device)
        
        return loss , loss.detach()
        
        
        
    
        
        
        
        