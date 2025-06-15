import ultralytics
from ultralytics import YOLO

from src.nn.tasks import (
    parse_model,
)

MODEL_NAMES = [
    "yolo8l-seg-pt",
]

def apply_custom_patches():
    """Apply custom monkey patches."""
    ultralytics.nn.tasks.parse_model = parse_model

def get_model(model_name, weights=None):
    """
    Get the appropriate model for a given model name.
    """
    
    # if model_name == "yolov8l-seg":
    #     model = YOLO("yolov8l-seg.yaml").load(weights) if weights else YOLO("yolov8l-seg.yaml")
    # elif model_name == "yolo11n-seg":
    #     model = YOLO("yolo11n-seg.yaml").load(weights) if weights else YOLO("yolo11n-seg.yaml")
    # elif model_name == "yolo11m-seg":
    #     model = YOLO("yolo11m-seg.yaml").load(weights) if weights else YOLO("yolo11m-seg.yaml")
    # elif model_name == "yolo11l-seg":
    #     model = YOLO("yolo11l-seg.yaml").load(weights) if weights else YOLO("yolo11l-seg.yaml")
    # elif model_name == "yolo11x-seg":
    #     model = YOLO("yolo11x-seg.yaml").load(weights) if weights else YOLO("yolo11x-seg.yaml")
    # else:
    #     raise ValueError(f"Unknown model name: {model_name}")
    
    if model_name in [
        "yolov8l-seg",
        "yolo11n-seg",
        "yolo11m-seg",
        "yolo11l-seg",
        "yolo11x-seg",
    ]:
        model = YOLO(weights) if weights else YOLO(f"{model_name}.yaml")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
