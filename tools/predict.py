# predict.py
import sys
import argparse
import json
import yaml
from pathlib import Path

import cv2
from ultralytics.utils import LOGGER

# Determine the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.model_registry import get_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict script for segmentation model inference."
    )
    
    parser.add_argument('--config-file', type=str, help='Path to a YAML or JSON configuration file containing inference parameters. These can be overridden by direct CLI args.')

    # Model and source arguments
    parser.add_argument("--model-name", type=str, help="Name of the model to use for prediction.")
    parser.add_argument("--weights", type=str, help="Path to the model weights file (e.g., .pt file).")
    parser.add_argument('--source', type=str, help='Input source (image/video/dir/URL/webcam)')

    # Inference arguments
    parser.add_argument('--conf', type=float, help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, help='Inference size (h,w)')
    parser.add_argument('--rect', action='store_true', help='Rectangular inference (minimal padding)')
    parser.add_argument('--half', action='store_true', help='FP16 quantization')
    parser.add_argument('--device', type=str, help='Device (cpu, cuda:0, ...)')
    parser.add_argument('--batch', type=int, help='Batch size (only for dir/video/txt)')
    parser.add_argument('--max-det', type=int, help='Maximum detections per image')
    parser.add_argument('--vid-stride', type=int, help='Video frame stride')
    parser.add_argument('--stream-buffer', action='store_true', help='Buffer video frames')
    parser.add_argument('--visualize', action='store_true', help='Visualize model features')
    parser.add_argument('--augment', action='store_true', help='Test-time augmentation')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class IDs')
    parser.add_argument('--retina-masks', action='store_true', help='High-res segmentation masks')
    parser.add_argument('--embed', nargs='+', type=int, help='Layer indices for feature extraction')
    parser.add_argument('--project', type=str, help='Project name for saving results')
    parser.add_argument('--name', type=str, help='Experiment name for saving results')
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.add_argument('--verbose', action='store_true', help='Show detailed logs')

    # Visualization arguments
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--save-frames', action='store_true', help='Save individual video frames')
    parser.add_argument('--save-txt', action='store_true', help='Save results as TXT')
    parser.add_argument('--save-conf', action='store_true', help='Save confidence scores')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped detections')
    parser.add_argument('--show-labels', action='store_true', help='Show labels')
    parser.add_argument('--show-conf', action='store_true', help='Show confidences')
    parser.add_argument('--show-boxes', action='store_true', help='Show bounding boxes')
    parser.add_argument('--line-width', type=int, help='Bounding box line width')

    return parser

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as file:
            config = json.load(file)
    else:
        raise ValueError("Unsupported configuration file format. Use .yaml or .json.")
    return config

def main():
    parser = parse_args()
    args = parser.parse_args()
    predict_config = {}

    # Load config file if provided
    if args.config_file:
        predict_config = load_config(args.config_file)
    else:
        predict_config = {}

    # Get parser defaults
    defaults = {key: parser.get_default(key) for key in vars(args)}

    # Only override config with CLI if CLI arg is different from default
    for key in vars(args):
        if key == 'config_file':
            continue
        cli_value = getattr(args, key)
        default_value = defaults[key]
        # For lists (like classes, embed), compare with None or default
        if isinstance(cli_value, list) or isinstance(default_value, list):
            if cli_value is not None:
                predict_config[key] = cli_value
        else:
            if cli_value != default_value and cli_value is not None:
                predict_config[key] = cli_value

    # Extract model_name and weights for model initialization
    model_name = predict_config.pop('model_name', None)
    weights = predict_config.pop('weights', None)

    if not model_name or not weights:
        raise ValueError("Both --model-name and --weights must be specified (either in config or as CLI args).")

    LOGGER.info(f"Using model: {model_name} with weights: {weights}")
    LOGGER.info(f"Predict configuration: {predict_config}")

    model = get_model(model_name, weights=weights)
    
    # Run prediction and process results
    results = model.predict(**predict_config)
    
    # Iterate through results to trigger inference/display
    for _ in results:
        pass  # Processing happens here

    # Keep windows open after prediction (if show=True)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
