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
    
    def __init__(self, model) -> None:
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
        
        id_masks = self.process_masks(batch)
        
        # Move mask to model device
        id_masks = id_masks.to(self.device)
        
        # Resize and ensure shape
        id_masks = F.interpolate(id_masks.float().unsqueeze(1), size=preds.shape[2:], mode='nearest').squeeze(1)
        id_masks = id_masks.long()
        
        # Sanity checks (optional)
        if preds.shape[0] != id_masks.shape[0] or preds.shape[2:] != id_masks.shape[1:]:
            raise ValueError("Shape mismatch between preds and mask.")

        loss = F.cross_entropy(preds, id_masks, reduction='mean', ignore_index=255)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN; check your data and model.")
        return loss, loss.detach()
    
    
    def process_masks(self, batch: dict, ignore_index: int = 255) -> torch.Tensor:
        """
        Returns new, processed masks without changing original input.
        - Background (0) → 255 (ignore)
        - Object indices (1,2,...) → actual class labels from cls
        Args:
            batch (dict): 
                'masks': list of 2D torch.Tensor masks (uint8/int)
                'cls': 1D tensor of class labels
                'batch_idx': 1D tensor mapping each class to its mask index
            ignore_index (int): value to use for background pixels (default: 255)
        Returns:
            list: processed masks, one per input mask, with updated values
        """
        masks = batch["masks"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"]
        processed = []

        for mask_idx, mask in enumerate(masks):
            mask_proc = mask.clone()                        # Copy, do not modify input
            mask_proc[mask_proc == 0] = ignore_index        # Background to ignore
            mask_cls = cls[batch_idx == mask_idx]           # Class labels for this mask
            map_length = int(mask_proc.max())
            mapping = torch.full((map_length + 1,), 255, dtype=mask_proc.dtype, device=mask_proc.device)
            for i, cl in enumerate(mask_cls, start=1):
                mapping[i] = cl                             # Map each index to class
            mask_remapped = mapping[mask_proc.long()]       # Vectorized remap
            processed.append(mask_remapped.type_as(mask))   # Keep original type

        return torch.stack(processed, dim=0)  # [B, H, W]
        
        
    
        
        
        
        