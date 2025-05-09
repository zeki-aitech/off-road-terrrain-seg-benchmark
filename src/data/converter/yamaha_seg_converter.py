from ultralytics.utils import LOGGER

from .base_converter import BaseConverter


class YamahaSegConverter(BaseConverter):
    
    """
    Converter for Yamaha Segmentation dataset.
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
        super().__init__(source_dir, output_dir, class_mapping)
    
    def convert(self):
        """
        Convert dataset to YOLO format
        """
        LOGGER.info(f"Converting Yamaha Segmentation dataset from {self.source_dir} to {self.output_dir}")
        