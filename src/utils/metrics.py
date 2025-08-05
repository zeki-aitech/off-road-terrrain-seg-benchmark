from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ultralytics.utils import metrics, LOGGER, DataExportMixin, SimpleClass, TryExcept, checks, plt_settings


class SemanticSegmentMetrics(SimpleClass, DataExportMixin):
    """
    Metrics class for semantic segmentation tasks.
    
    Attributes:
        mIoU (float): Mean Intersection over Union.
        speed (dict): A dictionary containing the time taken for each step in the pipeline.
        task (str): The task type, set to 'semantic_segment'.
    """
    
    def __init__(self, ignore_index: int = 255):
        """Initializes the SemanticSegmentMetrics instance."""
        self.miou = 0.0
        self.pixel_acc = 0.0
        self.mean_class_acc = 0.0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "semantic_segment"
        self.ignore_index = ignore_index  # Index to ignore in metrics calculations
        
    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """
        Process target classes and predicted classes to compute metrics.

        Args:
            targets (torch.Tensor): Target segmentation masks [H, W] or [N, H, W].
            pred (torch.Tensor): Predicted segmentation masks [H, W] or [N, H, W].
        """
        if isinstance(pred, list):
            # Check if all tensors have the same spatial dimensions
            if len(pred) > 1:
                shapes = [p.shape[-2:] for p in pred]
                if not all(shape == shapes[0] for shape in shapes):
                    # Resize all to the same size (take the largest dimensions)
                    max_h = max(shape[0] for shape in shapes)
                    max_w = max(shape[1] for shape in shapes)
                    
                    resized_pred = []
                    for p in pred:
                        if p.shape[-2:] != (max_h, max_w):
                            # Use interpolation for resizing
                            import torch.nn.functional as F
                            p_resized = F.interpolate(
                                p.float().unsqueeze(0), 
                                size=(max_h, max_w), 
                                mode='nearest'
                            ).squeeze(0).long()
                            resized_pred.append(p_resized)
                        else:
                            resized_pred.append(p)
                    pred = torch.cat(resized_pred)
                else:
                    pred = torch.cat(pred)
            else:
                pred = pred[0] if pred else torch.empty(0)
                
        if isinstance(targets, list):
            # Apply the same resizing logic to targets
            if len(targets) > 1:
                shapes = [t.shape[-2:] for t in targets]
                if not all(shape == shapes[0] for shape in shapes):
                    # Resize all to the same size (take the largest dimensions)
                    max_h = max(shape[0] for shape in shapes)
                    max_w = max(shape[1] for shape in shapes)
                    
                    resized_targets = []
                    for t in targets:
                        if t.shape[-2:] != (max_h, max_w):
                            # Use interpolation for resizing
                            import torch.nn.functional as F
                            t_resized = F.interpolate(
                                t.float().unsqueeze(0), 
                                size=(max_h, max_w), 
                                mode='nearest'
                            ).squeeze(0).long()
                            resized_targets.append(t_resized)
                        else:
                            resized_targets.append(t)
                    targets = torch.cat(resized_targets)
                else:
                    targets = torch.cat(targets)
            else:
                targets = targets[0] if targets else torch.empty(0)

        # Calculate mIoU
        self.miou = self.calculate_miou(pred, targets)
        # Calculate pixel accuracy
        self.pixel_acc = self.calculate_pixel_accuracy(pred, targets)
        # Calculate mean accuracy
        self.mean_class_acc = self.calculate_mean_class_accuracy(pred, targets)

    def calculate_pixel_accuracy(self, pred, targets):
        """Calculate pixel accuracy."""
        mask = (targets != self.ignore_index)
        correct = ((pred == targets) & mask).sum()
        total = mask.sum()
        return (correct.float() / total.float()).item()

    def calculate_mean_class_accuracy(self, pred, targets):
        """Calculate mean class accuracy."""
        classes = torch.unique(targets)
        class_accs = []

        for cls in classes:
            if cls == self.ignore_index:
                continue  # Skip ignore index class
            mask = (targets == cls)
            if mask.sum() > 0:
                correct = ((pred == cls) & mask).sum()
                class_acc = correct / mask.sum()
                class_accs.append(class_acc.item())

        return sum(class_accs) / len(class_accs) if class_accs else 0.0
        
    def calculate_miou(self, pred: torch.Tensor, targets: torch.Tensor, num_classes: int = None) -> float:
        """
        Calculate mean Intersection over Union (mIoU) for semantic segmentation.

        Args:
            pred (torch.Tensor): Predicted segmentation masks [N, H, W] or [H, W].
            targets (torch.Tensor): Ground truth segmentation masks [N, H, W] or [H, W].
            num_classes (int, optional): Number of classes. If None, inferred from data.

        Returns:
            float: Mean IoU score.
        """
        # Flatten tensors
        pred = pred.flatten()
        targets = targets.flatten()

        # Remove ignore index pixels
        valid_mask = targets != self.ignore_index
        pred = pred[valid_mask]
        targets = targets[valid_mask]

        # Get unique classes from targets (ground truth)
        if num_classes is None:
            classes = torch.unique(targets)
        else:
            classes = torch.arange(num_classes, device=targets.device)

        ious = []

        for cls in classes:
            # Create binary masks for current class
            pred_cls = (pred == cls)
            target_cls = (targets == cls)

            # Calculate intersection and union
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            # Calculate IoU for this class
            if union > 0:
                iou = intersection / union
                ious.append(iou)
            else:
                # If no pixels of this class exist in ground truth, skip it
                # (Some implementations add 1.0 for absent classes, others skip)
                continue
            
        # Return mean IoU
        return torch.stack(ious).mean().item() if ious else 0.0
        
    @property
    def fitness(self) -> float:
        """Calculate the fitness score based on mIoU."""
        return self.miou

    @property
    def results_dict(self) -> Dict[str, float]:
        """Return a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [self.miou, self.pixel_acc, self.mean_class_acc, self.fitness]))
    
    @property
    def keys(self) -> List[str]:
        """Return a list of keys for the results_dict property."""
        return ["metrics/mIoU", "metrics/pixel_accuracy", "metrics/mean_class_accuracy"]

    @property
    def curves(self) -> List:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> List:
        """Return a list of curves for accessing specific metrics curves."""
        return []
    
    def summary(self, normalize: bool = True, decimals: int = 5) -> List[Dict[str, float]]:
        """
        Generate a single-row summary of semantic segmentation metrics (mIoU).

        Args:
            normalize (bool): Whether to normalize the metrics.
            decimals (int): Number of decimal places to round the metrics.

        Returns:
            List[Dict[str, float]]: A list containing a single dictionary with the summary metrics.
        """
        return [{"mIoU": round(self.miou, decimals),
                 "pixel_acc": round(self.pixel_acc, decimals),
                 "mean_class_acc": round(self.mean_class_acc, decimals)}]
    
    
    
    
        
        
        
        

