from pathlib import Path
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from ultralytics.utils import LOGGER

from .base_converter import BaseConverter


YAMAHA_SEG_RGB_MAP = {
    (1, 88, 255): 0,    # blue - sky
    (156, 76, 30): 1,   # brown - rough trail
    (178, 176, 153): 2, # grey - smooth trail
    (255, 0, 128): 3,   # pink - slippery trail
    (128, 255, 0): 4,   # bright lime green - traversable grass
    (40, 80, 0): 5,     # dark green - high vegetation
    (0, 160, 0): 6,     # bright green - non-traversable low vegetation
    (255, 0, 0): 7,     # red - obstacle
}

YAMAHA_SEG_CLASSES = {
    0: "sky",
    1: "rough_trail",
    2: "smooth_trail",
    3: "slippery_trail",
    4: "traversable_grass",
    5: "high_vegetation",
    6: "non_traversable_low_vegetation",
    7: "obstacle",
}

class YamahaSegConverter(BaseConverter):
    
    """
    Converter for Yamaha Segmentation dataset.
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        class_mapping=YAMAHA_SEG_RGB_MAP
    ):
        """
        Args:
            source_dir (str): Directory containing the source data.
            output_dir (str): Directory to save the converted data.
            class_mapping (dict, optional): Mapping of class names to IDs. Defaults to None.
        """
        super().__init__(source_dir, output_dir, class_mapping)
        
        # self.pixel_to_class_mapping = \
        #     {i + 1: i for i in range(len(self.class_mapping))}
    
    def convert(self):
        """
        Convert dataset to YOLO format
        """
        LOGGER.info(f"Converting Yamaha Segmentation dataset from {self.source_dir} to {self.output_dir}")
        
        # Find all version directories (e.g., yamaha_v0)
        version_dirs = [d for d in self.source_dir.glob("yamaha_v*") if d.is_dir()]
        
        if not version_dirs:
            LOGGER.error(f"No version directories found in {self.source_dir}")
            return
        
        # Process each version directory
        for version_dir in version_dirs:
            version_name = version_dir.name
            LOGGER.info(f"Processing {version_name}...")
            
            # Process train and val splits
            for split in ["train", "valid"]:
                split_dir = version_dir / split

                if not split_dir.exists():
                    LOGGER.warning(f"Split directory {split_dir} does not exist. Skipping...")
                    continue
                
                # Get all ID folders
                id_folders = [d for d in split_dir.glob("iid*") if d.is_dir()]

                if not id_folders:
                    LOGGER.warning(f"No ID folders found in {split_dir}")
                    continue
                
                LOGGER.info(f"Found {len(id_folders)} ID folders in {split} split")
                
                # Create output directories
                images_output_dir = self.images_dir / (split if split == "train" else "val")
                labels_output_dir = self.labels_dir / (split if split == "train" else "val")
                images_output_dir.mkdir(parents=True, exist_ok=True)
                labels_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each ID folder
                for id_folder in tqdm(id_folders, desc=f"Converting {split} data"):
                    folder_id = id_folder.name
                    
                    # Check for rgb.jpg and labels.png
                    rgb_file = id_folder / "rgb.jpg"
                    mask_file = id_folder / "labels.png"
                    
                    if not rgb_file.exists() or not mask_file.exists():
                        continue
                    
                    # Copy image with a unique name based on folder ID
                    image_output_path = images_output_dir / f"{folder_id}.jpg"
                    shutil.copy(rgb_file, image_output_path)
                    
                    # Convert mask to YOLO format
                    yolo_format_label = self._convert_mask_to_yolo(mask_file)
                    if yolo_format_label is None:
                        LOGGER.warning(f"Failed to convert mask for {folder_id}, skipping...")
                        continue
                    
                    # Save YOLO format label
                    label_output_path = labels_output_dir / f"{folder_id}.txt"
                    with open(label_output_path, "w", encoding="utf-8") as file:
                        for item in yolo_format_label:
                            line = " ".join(map(str, item))
                            file.write(line + "\n")
                    LOGGER.info(f"Processed and stored at {label_output_path}")
                    
                    
        # Create YAML config
        self.create_yaml()   
        
        # LOGGER.info((f"Conversion complete. Output saved to {self.output_dir}"))
    
    def _convert_mask_to_yolo(self, mask_path):
        """
        Convert a color mask to YOLO segmentation format using grayscale processing
        """
        # if mask_path.suffix not in {".png", ".jpg"}:
        #     return None
        
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_height, img_width = mask.shape[:2]
        LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")
        
        unique_values = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        yolo_format_data = []
        
        # LOGGER.info(f"Unique values in mask: {unique_values}")

        for value in unique_values:
            if np.array_equal(value, np.array([255, 255, 255])):  # Background color
                continue  # Skip background
                
            class_index = self.class_mapping.get(tuple(value), -1)
            if class_index == -1:
                LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                continue
        
            # Create a binary mask for the current class and find contours
            contours, _ = cv2.findContours(
                np.all(mask == value, axis=-1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # Find contours

            for contour in contours:
                if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                    contour = contour.squeeze()  # Remove single-dimensional entries
                    yolo_format = [class_index]
                    for point in contour:
                        # Normalize the coordinates
                        yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                        yolo_format.append(round(point[1] / img_height, 6))
                    yolo_format_data.append(yolo_format)
        
        return yolo_format_data
        
        

# if __name__ == "__main__":
    # test the converter
        