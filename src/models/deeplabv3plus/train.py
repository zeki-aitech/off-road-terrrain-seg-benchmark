from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG


class DeepLabV3PlusTrainer(yolo.segment.SegmentationTrainer):
    """
    Trainer for the DeepLabV3+ model.
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the DeepLabV3PlusTrainer with the given configuration.
        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)