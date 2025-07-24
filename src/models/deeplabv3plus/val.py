
from copy import copy
from typing import Dict, Any

import torch
from ultralytics.models import yolo

from src.utils.metrics import SemanticSegmentMetrics

class DeepLabV3PlusSemanticSegmentationValidator(yolo.segment.SegmentationValidator):
    """
    Validator for the DeepLabV3+ model.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task
        self.metrics = SemanticSegmentMetrics()
        
        
    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "mIoU",
            "Pixel Accuracy",
            "Mean Class Accuracy"
        )
        
    def init_metrics(self, model: torch.nn.Module) -> None:
        
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        # batch["cls"] = batch["cls"].to(self.device)
        # batch["masks"] = batch["masks"].to(self.device).float()
        return batch
        
    def update_metrics(self, preds: torch.Tensor, batch: Dict[str, Any]) -> None:
        """
        Update running metrics with model predictions and batch targets.

        Args:
            preds (torch.Tensor): Model predictions.
            batch (Dict[str, Any]): Batch data containing images and labels.
        """
        self.pred.append(preds)
        self.targets.append(batch["targets"])

    def get_stats(self) -> Dict[str, float]:
        """Calculate and return a dictionary of metrics by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict
    
    