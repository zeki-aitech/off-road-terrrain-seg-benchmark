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
    
    def __init__(self) -> None:
        """Initializes the SemanticSegmentMetrics instance."""
        self.mIoU = 0.0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "semantic_segment"
        
    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """
        Process target classes and predicted classes to compute metrics.

        Args:
            targets (torch.Tensor): Target classes.
            pred (torch.Tensor): Predicted classes.
        """
        pred, targets = torch.cat(pred), torch.cat(targets)
        
    
    @property
    def keys(self) -> List[str]:
        """Return a list of keys for the results_dict property."""
        return ["metrics/mIoU",]

    @property
    def curves(self) -> List:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> List:
        """Return a list of curves for accessing specific metrics curves."""
        return []
    
    
    
    
        
        
        
        

