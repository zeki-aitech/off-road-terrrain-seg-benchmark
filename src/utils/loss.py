from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER, colorstr
from src.utils.mask_processing import convert_instance_masks_to_semantic


class DeepLabV3PlusSemanticSegmentationLoss:
    """
    DeepLabV3+ loss function for semantic segmentation.
    
    This class implements the loss function used in the DeepLabV3+ model, following
    Ultralytics patterns for loss computation and device handling.
    
    Attributes:
        device (torch.device): Device where computations will be performed.
        
    Examples:
        >>> model = DeepLabV3PlusModel()
        >>> loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        >>> preds = torch.randn(2, 20, 640, 640)
        >>> batch = {"masks": [...], "cls": [...], "batch_idx": [...]}
        >>> loss, loss_items = loss_fn(preds, batch)
    """
    
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initialize the DeepLabV3PlusLoss.
        
        Args:
            model (torch.nn.Module): The model instance to extract device information.
        """
        self.device = next(model.parameters()).device
        LOGGER.info(f"{colorstr('Loss:')} Initialized DeepLabV3+ segmentation loss on {self.device}")
        
    def __call__(
        self,
        preds: torch.Tensor, 
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the DeepLabV3+ model.

        Args:
            preds (torch.Tensor): Model predictions with shape [B, C, H, W] where C is number of classes.
            batch (Dict[str, Any]): Batch data containing input images and target masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Total loss and detached loss for logging.
            
        Raises:
            ValueError: If shapes don't match or loss is NaN.
        """
        # Convert instance masks to semantic segmentation format
        semantic_masks = convert_instance_masks_to_semantic(batch)
        
        # Move mask to model device
        semantic_masks = semantic_masks.to(self.device, non_blocking=True)
        
        # Resize mask to match prediction spatial dimensions if needed
        if semantic_masks.shape[-2:] != preds.shape[-2:]:
            semantic_masks = F.interpolate(
                semantic_masks.float().unsqueeze(1), 
                size=preds.shape[2:], 
                mode='nearest'
            ).squeeze(1).long()
        else:
            semantic_masks = semantic_masks.long()
        
        # Validate shapes
        if preds.shape[0] != semantic_masks.shape[0]:
            raise ValueError(f"Batch size mismatch: preds {preds.shape[0]} vs masks {semantic_masks.shape[0]}")
        if preds.shape[2:] != semantic_masks.shape[1:]:
            raise ValueError(f"Spatial size mismatch: preds {preds.shape[2:]} vs masks {semantic_masks.shape[1:]}")

        # Compute cross-entropy loss with ignore index for background
        loss = F.cross_entropy(
            preds, 
            semantic_masks, 
            reduction='mean', 
            ignore_index=255
        )
        
        # Check for NaN loss
        if torch.isnan(loss):
            # Check if all targets are ignore_index (empty batch case)
            valid_pixels = (semantic_masks != 255).sum()
            if valid_pixels == 0:
                # If all pixels are ignored, return a zero loss
                LOGGER.warning(f"{colorstr('yellow', 'Warning:')} All pixels ignored, returning zero loss")
                loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
            else:
                # Real NaN error - should raise exception
                LOGGER.error(f"{colorstr('red', 'Error:')} Loss is NaN - check your data and model")
                raise ValueError("Loss is NaN; check your data and model.")
            
        # Return loss and detached version for logging (following Ultralytics pattern)
        return loss, loss.detach()