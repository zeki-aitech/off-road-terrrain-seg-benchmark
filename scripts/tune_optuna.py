import sys
from pathlib import Path
from copy import deepcopy

import json
import yaml

import optuna
import ultralytics
import ultralytics.engine
import ultralytics.engine.model
from ultralytics.utils import LOGGER

# Determine the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.models.model_registry import get_model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a model with Optuna hyperparameter optimization."
    )
    parser.add_argument('--config-file', type=str, default=None, help='Path to a YAML or JSON config file.')
    parser.add_argument("--study-name", type=str, help="Name of the Optuna study.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--model-name", type=str, help="Name of the model to train.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file.")
    parser.add_argument("--data", type=str, help="Path to the data config file.")
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
    return parser.parse_args()

def seg_objective(trial, model, config):
    full_params = deepcopy(config)
    # 1. Suggest hyperparameters using Optuna
    tune_params = {}
    for param, value in full_params.get('tune', {}).items():
        suggest_type = value[0]
        suggest_args = value[1].copy()
        # Convert YAML null/"None" to Python None
        suggest_args = [
            None if (isinstance(arg, str) and arg.lower() in ["none", "null"]) else arg
            for arg in suggest_args
        ]
        # Handle 'log' for float/int
        log_flag = False
        if suggest_type in ('float', 'int') and len(suggest_args) > 3 and isinstance(suggest_args[3], bool):
            log_flag = suggest_args[3]
            suggest_args = suggest_args[:3]
        if suggest_type == 'float':
            step = suggest_args[2] if len(suggest_args) > 2 else None
            if log_flag and step is not None:
                step = None
            tune_params[param] = trial.suggest_float(
                name=param,
                low=float(suggest_args[0]),
                high=float(suggest_args[1]),
                step=step,
                log=log_flag,
            )
        elif suggest_type == 'int':
            step = suggest_args[2] if len(suggest_args) > 2 else 1
            if log_flag and step != 1:
                log_flag = False
            tune_params[param] = trial.suggest_int(
                name=param,
                low=int(suggest_args[0]),
                high=int(suggest_args[1]),
                step=step,
                log=log_flag,
            )
        elif suggest_type == 'categorical':
            # Convert YAML lists to tuples for hashability, if needed
            choices = [tuple(x) if isinstance(x, list) else x for x in suggest_args]
            tune_params[param] = trial.suggest_categorical(
                name=param,
                choices=choices,
            )
    full_params.update(tune_params)
    full_params.pop('tune', None)
    LOGGER.info(f"Training with parameters: {full_params}")
    # 2. Train the model
    model.train(**full_params)
    # 3. Evaluate the model's performance
    metrics = model.val(
        data=config.get('data'),
        imgsz=config.get('imgsz'),
    )
    return metrics.seg.map

def main():
    args = parse_args()
    opt_config = {}
    model_name = None
    model_weights = None
    study_name = None
    n_trials = None

    # Load configuration from file if provided
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                if args.config_file.endswith((".yaml", ".yml")):
                    config_params = yaml.safe_load(f)
                elif args.config_file.endswith(".json"):
                    config_params = json.load(f)
                else:
                    LOGGER.warning(f"Unknown config file format for {args.config_file}. Attempting to load as YAML.")
                    config_params = yaml.safe_load(f)
                if config_params:
                    if 'model_name' in config_params:
                        model_name = config_params.pop('model_name')
                    if 'weights' in config_params:
                        model_weights = config_params.pop('weights')
                    if 'study_name' in config_params:
                        study_name = config_params.pop('study_name')
                    if 'n_trials' in config_params:
                        n_trials = config_params.pop('n_trials')
                    opt_config.update(config_params)
                LOGGER.info(f"Loaded training parameters from config file: {args.config_file}")
        except FileNotFoundError:
            LOGGER.error(f"Configuration file not found: {args.config_file}")
            sys.exit(1)
        except Exception as e:
            LOGGER.error(f"Error loading or parsing configuration file {args.config_file}: {e}", exc_info=True)
            sys.exit(1)

    # Overridable arguments
    if args.study_name is not None: study_name = args.study_name
    if args.n_trials is not None: n_trials = args.n_trials
    if args.model_name is not None: model_name = args.model_name
    if args.weights is not None: model_weights = args.weights
    if args.data is not None: opt_config['data'] = args.data

    LOGGER.info("-" * 30)
    LOGGER.info(f"Study name: {study_name}")
    LOGGER.info(f"Model name: {model_name}")
    LOGGER.info(f"Model weights: {model_weights}")
    LOGGER.info("Parameters:")
    for key, value in sorted(opt_config.items()):
        LOGGER.info(f"  {key}: {value}")
    LOGGER.info("-" * 30)

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name
    )

    study.optimize(
        lambda trial: seg_objective(
            trial,
            model=get_model(model_name, weights=model_weights),
            config=opt_config
        ),
        n_trials=n_trials or 50
    )

if __name__ == "__main__":
    main()
