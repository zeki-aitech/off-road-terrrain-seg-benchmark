"""
Interface for DeepLabV3+ model.
"""
from typing import Dict, Any

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import SegmentationModel

from src.nn.tasks import (
    DeepLabV3PlusSemanticSegmentationModel
)

from .train import DeepLabV3PlusSemanticSegmentationTrainer
from .val import DeepLabV3PlusSemanticSegmentationValidator
from .predict import DeepLabV3PlusSemanticSegmentationPredictor

class DeepLabV3Plus(Model):

    def __init__(self, model="deeplabv3plus.yaml", task="segment", verbose=False):
        """
        Initializes the DeepLabV3Plus model.
        Args:
            model (str): Path to the model configuration file or model name.
            task (str): Task type, default is "segmentation".
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)
        
    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            "segment": {
                "model": DeepLabV3PlusSemanticSegmentationModel,
                "trainer": DeepLabV3PlusSemanticSegmentationTrainer,
                "validator": DeepLabV3PlusSemanticSegmentationValidator,
                "predictor": DeepLabV3PlusSemanticSegmentationPredictor,
            },
        }