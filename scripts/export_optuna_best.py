import argparse
import os
import optuna
from ruamel.yaml import YAML

def export_best_params(storage_path, output_yaml='best_params.yaml', input_yaml=None):
    # Prepare Optuna storage URL
    storage_url = f'sqlite:///{os.path.abspath(storage_path)}'
    study = optuna.load_study(
        study_name=None,
        storage=storage_url)
    best_params = study.best_params
    best_value = study.best_value

    yaml = YAML()
    yaml.preserve_quotes = True

    if input_yaml:
        # Load existing config and update/add best params for keys in 'tune'
        with open(input_yaml, 'r') as f:
            config = yaml.load(f)
        tune_params = config.get('tune', {}).keys()
        for param in tune_params:
            if param in best_params:
                config[param] = best_params[param]
        output_data = config
    else:
        # No input: just export all best params
        output_data = best_params

    with open(output_yaml, 'w') as f:
        f.write(f"# best_value: {best_value}\n")
        yaml.dump(output_data, f)
    print(f"Best parameters exported to {output_yaml}")

def main():
    parser = argparse.ArgumentParser(description='Export Optuna best parameters to YAML')
    parser.add_argument('--input', type=str, required=False, help='Input YAML config file (optional)')
    parser.add_argument('--storage', type=str, required=True, help='Path to Optuna SQLite database')
    parser.add_argument('--output', type=str, default='best_params.yaml', help='Output YAML file')
    args = parser.parse_args()
    
    export_best_params(args.storage, args.output, args.input)

if __name__ == '__main__':
    main()
