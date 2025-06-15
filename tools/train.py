import os
import sys
import argparse
import json
import yaml 
import shutil 
from pathlib import Path
from ultralytics.utils import LOGGER

from ultralytics import settings

# Determine the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project root to the Python path
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.model_registry import get_model

def get_args():
    parser = argparse.ArgumentParser(description="Train a segmentation or detection model using Ultralytics.")

    # Configuration file for bulk training parameters
    parser.add_argument(
        '--config-file',
        type=str,
        default=None,
        help='Path to a YAML or JSON configuration file containing training parameters. These can be overridden by --extra or direct CLI args.'
    )
    
    parser.add_argument(
        '--tensorboard', action='store_true', help='Enable TensorBoard logging.'
    )
    parser.add_argument(
        '--mlflow', action='store_true', help='Enable MLflow logging.'
    )

    # Direct CLI arguments for common training parameters (override config-file and extra)
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the model to train.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the model weights file (e.g., .pt file).",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data configuration file (e.g., data.yaml for Ultralytics).",
    )
    parser.add_argument(
        '--epochs', type=int, help='Override number of epochs to train the model.',
    )
    parser.add_argument(
        '--batch-size', type=int, help='Override batch size for training. Use -1 for auto-batch.',
    )
    parser.add_argument(
        '--imgsz', type=int or list, help='Override input image size for training.',
    )
    parser.add_argument(
        '--learning-rate', type=float, help='Override initial learning rate (lr0).'
    )
    parser.add_argument(
        "--project", type=str, help="Override project name/directory to save training results.",
    )
    parser.add_argument(
        "--name", type=str, help="Override experiment name (subdirectory within project).",
    )
    parser.add_argument(
        '--device', type=str, help="Override device to train on (e.g., 'cpu', '0', '0,1').",
    )
    parser.add_argument(
        '--workers', type=int, help="Override number of worker threads for data loading."
    )
    # JSON string for additional, less common parameters (override config-file, overridden by direct CLI args)
    parser.add_argument(
        '--extra-params', # Renamed for clarity from just '--extra'
        type=str, 
        default='{}', # Default to an empty JSON object string
        help='Additional training parameters in JSON format (e.g., \'{"patience": 50, "optimizer": "AdamW"}\'). '
             'These override --config-file but are overridden by direct CLI arguments. '
             'See Ultralytics train settings for available keys: https://docs.ultralytics.com/modes/train/#train-settings'
    )

    return parser.parse_args()

def main():
    args = get_args()

    # --- 1. Initialize training parameters ---
    train_kwargs = {}
    model_name = None
    model_weights = None
    
    # --- 2. Load from --config-file (Lowest precedence after defaults) ---
    config_file_path_to_save = None
    if args.config_file:
        config_file_path_to_save = args.config_file # Store for later copying
        try:
            with open(args.config_file, 'r') as f:
                if args.config_file.endswith((".yaml", ".yml")):
                    config_params = yaml.safe_load(f)
                elif args.config_file.endswith(".json"):
                    config_params = json.load(f)
                else:
                    LOGGER.warning(f"Unknown config file format for {args.config_file}. Attempting to load as YAML.")
                    config_params = yaml.safe_load(f)
                
                if config_params: # Ensure it's not None or empty
                    if 'model_name' in config_params:
                        model_name = config_params.pop('model_name')
                    if 'weights' in config_params:
                        model_weights = config_params.pop('weights')
                    train_kwargs.update(config_params)
                LOGGER.info(f"Loaded training parameters from config file: {args.config_file}")
        except FileNotFoundError:
            LOGGER.error(f"Configuration file not found: {args.config_file}")
            sys.exit(1)
        except Exception as e:
            LOGGER.error(f"Error loading or parsing configuration file {args.config_file}: {e}", exc_info=True)
            sys.exit(1)

    # --- 3. Load from --extra-params JSON string (Overrides --config-file) ---
    if args.extra_params:
        try:
            extra_args_dict = json.loads(args.extra_params)
            train_kwargs.update(extra_args_dict)
            LOGGER.info(f"Loaded extra parameters from --extra-params JSON: {extra_args_dict}")
        except json.JSONDecodeError as e:
            LOGGER.error(f"Invalid JSON string provided for --extra-params: {args.extra_params}. Error: {e}")
            sys.exit(1)

    # --- 4. Apply direct CLI arguments (Highest precedence) ---

    # Overridable arguments
    if args.model_name is not None: model_name = args.model_name
    if args.weights is not None: model_weights = args.weights
    if args.data is not None: train_kwargs['data'] = args.data
    if args.epochs is not None: train_kwargs['epochs'] = args.epochs
    if args.batch_size is not None: train_kwargs['batch'] = args.batch_size # Ultralytics uses 'batch'
    if args.imgsz is not None: train_kwargs['imgsz'] = args.imgsz
    if args.learning_rate is not None: train_kwargs['lr0'] = args.learning_rate # Ultralytics uses 'lr0'
    if args.project is not None: train_kwargs['project'] = args.project
    if args.name is not None: train_kwargs['name'] = args.name
    if args.device is not None: train_kwargs['device'] = args.device
    if args.workers is not None: train_kwargs['workers'] = args.workers

    # Set defaults if not provided by any means (config, extra, or direct CLI)
    # Ultralytics train() has its own defaults, so we only set critical ones or ones for our script's logic
    # e.g. if we want a script-level default for 'epochs' if not specified anywhere.
    # For this example, we rely on Ultralytics defaults if not set.

    LOGGER.info(f"Final training parameters after merging all sources:")
    LOGGER.info(f" Model Name: {model_name}")
    LOGGER.info(f" Model Weights: {model_weights if model_weights else 'Default weights will be used.'}")
    for key, value in sorted(train_kwargs.items()): # Sorted for consistent logging
        LOGGER.info(f"  {key}: {value}")
    LOGGER.info("-" * 30)
    
    # --- logging settings ---
    if args.tensorboard:
        settings.update({"tensorboard": True})
    if args.mlflow:
        settings.update({"mlflow": True})
        # export environment variables for MLflow
        # os.environ["MLFLOW_EXPERIMENT_NAME"] = train_kwargs.get("project", "default")
        # os.environ["MLFLOW_RUN"] = train_kwargs.get("name", "default")

    # --- Model Loading ---
    try:
        LOGGER.info(f"Loading model: {model_name}...")
        model = get_model(model_name, weights=model_weights)
        LOGGER.info(f"Model '{model_name}' loaded successfully.")
    except ValueError as e:
        LOGGER.error(f"Error loading model: {e}")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred while loading the model: {e}", exc_info=True)
        sys.exit(1)

    # --- Model Training ---
    LOGGER.info(f"\nStarting training...")
    try:
        results = model.train(**train_kwargs)
        
        LOGGER.info("\nTraining completed successfully!")
        save_dir = results.save_dir if hasattr(results, 'save_dir') \
            else os.path.join(train_kwargs.get('project', 'runs/train'), train_kwargs.get('name', 'exp'))
        LOGGER.info(f"Results saved in: {save_dir}")

        # --- 5. Copy the used config file to the results directory for reproducibility ---
        if config_file_path_to_save and os.path.exists(save_dir):
            try:
                destination_config_path = os.path.join(save_dir, os.path.basename(config_file_path_to_save))
                shutil.copy2(config_file_path_to_save, destination_config_path)
                LOGGER.info(f"Copied training config file '{config_file_path_to_save}' to '{destination_config_path}'")
            except Exception as e:
                LOGGER.warning(f"Could not copy config file to results directory: {e}")
        
        # Log metrics if available
        if hasattr(results, 'metrics') and results.metrics:
            LOGGER.info("Training metrics:")
            for metric, value in results.metrics.items():
                LOGGER.info(f"  {metric}: {value}") # Corrected line

    except AttributeError as e: # This is the start of the exception handling for the training try block
        LOGGER.error(f"Error: The loaded model may not have a 'train' method or it's incompatible: {e}", exc_info=True)
        LOGGER.error("Please ensure the model retrieved by 'get_model' is a trainable Ultralytics model object.")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
