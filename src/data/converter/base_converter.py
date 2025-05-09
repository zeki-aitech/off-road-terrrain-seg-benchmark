from abc import ABC, abstractmethod
from pathlib import Path

class BaseConverter(ABC):
    
    """
    Abstract base class for data converters.
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        class_mapping=None
    ):
        """
        Args:
            source_dir (str): Directory containing the source data.
            output_dir (str): Directory to save the converted data.
            class_mapping (dict, optional): Mapping of class names to IDs. Defaults to None.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping if class_mapping else {}

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        for split in ["train", "val"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def convert(self):
        """
        Convert dataset to YOLO format
        """
        pass
    
    def copy_images(self, source_images_dir, target_split="train"):
        """
        Copy images to the output directory.
        """
        pass
        
    
    def create_yaml(self):
        """
        Create YAML config for converted dataset.
        """
        pass