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
            
            # Convert to numpy
            semantic_mask = pred_classes.cpu().numpy().astype(np.uint8)
            
            # Create Results object
            # For semantic segmentation, we store the mask directly as a 2D array
            result = Results(
                orig_img=orig_img,
                path=getattr(self, 'source', ''),
                names=getattr(self.model, 'names', {}),
                masks=semantic_mask,  # Store as 2D array [H, W]
                boxes=None,  # No bounding boxes for semantic segmentation
                probs=None   # No classification probabilities
            )
            
            results.append(result)
            
        return results
    
    def write_results(self, idx: int, results: List[Results], batch: Dict[str, Any]) -> str:
        """
        Write semantic segmentation results to file.
        
        Args:
            idx (int): Index of the image in the batch
            results (List[Results]): Prediction results
            batch (Dict[str, Any]): Batch information
            
        Returns:
            str: Log string for this prediction
        """
        p, im, _ = batch
        log_string = ""
        
        if len(results):
            result = results[0]
            if result.masks is not None:
                # Get unique classes in the mask (excluding background)
                mask = result.masks if isinstance(result.masks, np.ndarray) else result.masks[0]
                unique_classes = np.unique(mask)
                unique_classes = unique_classes[unique_classes > 0]  # Remove background
                
                log_string += f"{len(unique_classes)} classes detected: "
                if hasattr(self.model, 'names'):
                    class_names = [self.model.names.get(int(cls), f"class_{cls}") 
                                 for cls in unique_classes]
                    log_string += ", ".join(class_names)
                else:
                    log_string += ", ".join([f"class_{cls}" for cls in unique_classes])
                    
                # Save semantic mask if save is enabled
                if self.args.save or self.args.save_txt:
                    save_path = Path(self.save_dir) / f"{Path(p).stem}_semantic_mask.png"
                    # Save mask as image (pixel values = class IDs)
                    import cv2
                    cv2.imwrite(str(save_path), mask)
                    log_string += f" -> saved to {save_path}"
        
        return log_string
        