
from copy import copy

from ultralytics.models import yolo

class DeepLabV3PlusSemanticSegmentationValidator(yolo.segment.SegmentationValidator):
    """
    Validator for the DeepLabV3+ model.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        """
        super().__init__(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )