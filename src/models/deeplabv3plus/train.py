# src/models/deeplabv3plus/train.py

from copy import copy

# Apply monkey patches before importing Ultralytics
from src.patches import apply_patches
apply_patches()

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG, RANK

from src.nn.tasks import (
    DeepLabV3PlusSemanticSegmentationModel
)

from .val import DeepLabV3PlusSemanticSegmentationValidator


class DeepLabV3PlusSemanticSegmentationTrainer(yolo.segment.SegmentationTrainer):
    """
    Trainer for the DeepLabV3+ semantic segmentation model.
    
    This trainer extends the Ultralytics SegmentationTrainer to handle semantic segmentation
    tasks using DeepLabV3+ architecture while maintaining full compatibility with 
    Ultralytics instance segmentation datasets.
    
    Attributes:
        loss_names (List[str]): Names of the loss functions used during training.
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the DeepLabV3PlusSemanticSegmentationTrainer with the given configuration.
        
        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        # Initialize loss_names before calling parent constructor
        self.loss_names = ["loss"]
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return a DeepLabV3Plus model with specified configuration and weights.
        
        Args:
            cfg (str | dict, optional): Model configuration file path or dictionary.
            weights (str, optional): Path to model weights file.
            verbose (bool): Whether to display model information.
            
        Returns:
            DeepLabV3PlusSemanticSegmentationModel: Initialized model ready for training.
        """
        # Use default config if none provided
        if cfg is None:
            cfg = "/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml"
            
        model = DeepLabV3PlusSemanticSegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model

    def progress_string(self) -> str:
        """Return a formatted string showing training progress for semantic segmentation."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Classes",
            "Size",
        )
    
    def get_validator(self):
        """Return an instance of DeepLabV3PlusSemanticSegmentationValidator for validation."""
        # Use test_loader if available (during training), otherwise use None
        test_loader = getattr(self, 'test_loader', None)
        return DeepLabV3PlusSemanticSegmentationValidator(
            test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.
        
        Args:
            loss_items (torch.Tensor, optional): Tensor containing loss values.
            prefix (str): Prefix for loss item keys.
            
        Returns:
            dict | list: Dictionary with labeled loss items if loss_items provided, otherwise list of keys.
        """
        # For semantic segmentation, we typically have a single loss
        keys = [f"{prefix}/loss"]
        if loss_items is not None:
            loss_items = [round(float(loss_items), 5)]
            return dict(zip(keys, loss_items))
        else:
            return keys


