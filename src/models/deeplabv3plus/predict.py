import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Any, Dict

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.engine.results import Results


class DeepLabV3PlusSemanticSegmentationPredictor(yolo.segment.SegmentationPredictor):
    """
    Predictor for the DeepLabV3+ semantic segmentation model.
    
    This predictor handles semantic segmentation output, converting the model's
    class probability maps into semantic segmentation masks.
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the DeepLabV3PlusPredictor with the given configuration.
        Args:
            cfg (dict): Configuration dictionary with default prediction settings.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during prediction.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        
    def postprocess(self, preds: torch.Tensor, img: torch.Tensor, orig_imgs: List[np.ndarray]) -> List[Results]:
        """
        Post-process DeepLabV3+ predictions to create semantic segmentation results.
        
        Args:
            preds (torch.Tensor): Model predictions with shape [B, C, H, W] (class logits)
            img (torch.Tensor): Input images tensor
            orig_imgs (List[np.ndarray]): Original input images
            
        Returns:
            List[Results]: List of Results objects containing semantic segmentation masks
        """
        # Handle the case where preds might be a list/tuple first
        if isinstance(preds, (list, tuple)):
            preds = preds[0]  # Take the first element (main prediction)
            
        # Convert to tensor if needed
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        
        # Ensure tensor is on correct device
        device = getattr(self, 'device', 'cpu')
        if hasattr(preds, 'to'):
            preds = preds.to(device)
            
        # Ensure we have the right dimensions [B, C, H, W]
        if preds.dim() == 3:  # [C, H, W] -> [1, C, H, W]
            preds = preds.unsqueeze(0)
        elif preds.dim() == 2:  # [H, W] -> [1, 1, H, W]
            preds = preds.unsqueeze(0).unsqueeze(0)
            
        batch_size = preds.shape[0]
        results = []
        
        # Ensure orig_imgs is a list
        if not isinstance(orig_imgs, list):
            orig_imgs = [orig_imgs]
        
        for i in range(batch_size):
            # Get prediction for this image
            pred = preds[i]  # [C, H, W]
            orig_img = orig_imgs[i] if i < len(orig_imgs) else orig_imgs[0]
            
            # Convert logits to class predictions
            if pred.shape[0] > 1:  # Multi-class
                # Apply softmax to get probabilities
                pred_probs = F.softmax(pred, dim=0)  # [C, H, W]
                # Get class predictions
                pred_classes = torch.argmax(pred_probs, dim=0)  # [H, W]
            else:
                # Binary case
                pred_classes = (torch.sigmoid(pred.squeeze(0)) > 0.5).long()
                
            # Resize mask to original image size
            orig_h, orig_w = orig_img.shape[:2]
            if pred_classes.shape != (orig_h, orig_w):
                # Resize using nearest neighbor to preserve class labels
                pred_classes = F.interpolate(
                    pred_classes.unsqueeze(0).unsqueeze(0).float(),
                    size=(orig_h, orig_w),
                    mode='nearest'
                ).squeeze().long()
            
            # Convert to numpy for final result, but keep tensor for Results object
            semantic_mask_np = pred_classes.cpu().numpy().astype(np.uint8)
            # Keep as tensor for Results object (expected format)
            semantic_mask_tensor = pred_classes.unsqueeze(0)  # Add batch dim: [1, H, W]
            
            # Create Results object
            # For semantic segmentation, we store the mask as tensor
            # We need to provide empty tensors for boxes to avoid plotting errors
            empty_boxes = torch.empty(0, 6)  # Empty boxes tensor [N, 6] format
            
            result = Results(
                orig_img=orig_img,
                path=getattr(self, 'source', ''),
                names=getattr(self.model, 'names', {}),
                masks=semantic_mask_tensor,  # Store as tensor [1, H, W]
                boxes=empty_boxes,    # Empty boxes tensor instead of None
                probs=None   # No classification probabilities
            )
            
            results.append(result)
            
        return results
        