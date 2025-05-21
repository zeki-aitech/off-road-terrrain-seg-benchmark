import ultralytics
from ultralytics import YOLO, FastSAM

from src.nn.tasks import (
    parse_model,
)

MODEL_NAMES = [
    "yolo8l-seg-pt",
    "fatsam-s-pt",
]

def apply_custom_patches():
    """Apply custom monkey patches."""
    ultralytics.nn.tasks.parse_model = parse_model

def get_model(model_name, weights=None):
    """
    Get the appropriate model for a given model name.
    """
    
    if model_name == "yolov8l-seg":
        model = YOLO("yolov8l-seg.yaml").load(weights) if weights else YOLO("yolov8l-seg.yaml")
    elif model_name == "yolo11n-seg":
        model = YOLO("yolo11n-seg.yaml").load(weights) if weights else YOLO("yolo11n-seg.yaml")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
