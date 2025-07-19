
from copy import copy

from ultralytics.models import yolo

class DeepLabV3PlusSemanticSegmentationValidator(yolo.segment.SegmentationValidator):
    """
    Validator for the DeepLabV3+ model.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        
        
    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "mIoU",
            "Pixel Accuracy",
            "Mean Class Accuracy"
        )
        
    def init_metrics(self, model):
        return super().init_metrics(model)
        
    