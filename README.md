# Off-Road Terrain Segmentation Benchmark

## Scripts
### 1. Convert Dataset
The original datasets used in this study (e.g., KITTI, Yamaha-CMU Off-Road) often come in various proprietary or specific formats. To be used with the models in this study, these datasets first need to be converted into a standardized annotation format.

The `scripts/convert_dataset.py` script is provided to facilitate this conversion process. This script processes the original dataset files and reorganizes them into the required structure and annotation style.

**Usage:**

To perform the conversion, run the script from your project's root directory using the following command structure. You will need to specify an identifier for the dataset converter to use, the source directory of the original dataset, the desired output directory for the converted files, and any extra arguments specific to that particular converter.

```
python3 scripts/convert_dataset.py
--dataset <dataset_name>
--source-dir <path_to_original_dataset>
--output-dir <path_for_converted_dataset>
--extra-args '<json_formatted_extra_arguments>'
```

**Placeholders:**
*   `<dataset_name>`: Identifier for the specific dataset converter to use (e.g., `yamaha_seg` is for YAMAHA-CMU Off-Road Dataset).
*   `<path_to_original_dataset>`: The file path to the directory containing the original, unconverted dataset.
*   `<path_for_converted_dataset>`: The file path where the standardized, converted dataset will be saved.
*   `'<json_formatted_extra_arguments>'`: Optional. Additional parameters for the specific dataset converter, provided as a JSON string enclosed in single quotes (e.g., `'{"key": "value"}'`). These arguments allow for customized conversion behavior.

**Example:**

To convert the Yamaha-CMU Off-Road dataset, potentially for a segmentation task, and specify a `min_contour_pixel_area` of 300 using `extra-args`:

```
python3 scripts/convert_dataset.py
--dataset yamaha_seg
--source-dir datasets/origin/yamaha_seg
--output-dir datasets/converted/yamaha_seg
--extra-args '{"min_contour_pixel_area":300}'
```

Ensure the paths for `--source-dir` and `--output-dir` are correct for your local file system. The specific output format will depend on the implementation of the converter chosen via the `--dataset` argument.


### 2. Browse Dataset
The `scripts/browse_dataset.py` script allows you to visually inspect images and their corresponding segmentation labels from a dataset configured with a `dataset.yaml` or `data.yaml` file (YOLO format). It displays images with overlaid segmentation polygons and class names.

**Usage:**

Run the script from your project's root directory:
```
python3 scripts/browse_dataset.py --dataset <path_to_dataset_root> [--split <split_name>]
```

**Arguments:**

*   **`--dataset <path_to_dataset_root>`** (Required):
    *   Path to the root directory of your dataset. This directory must contain a `dataset.yaml` or `data.yaml` file that defines the dataset structure (paths to images/labels, class names, and splits like 'train', 'val', 'test').
*   **`--split <split_name>`** (Optional):
    *   Specify which dataset split to browse (e.g., `train`, `val`, `test`).
    *   If not provided, it defaults to 'train' if available, then 'val'. If neither is found or multiple splits exist without a clear default, the script will prompt you to choose from the available splits defined in the YAML file.

**How it Works:**

1.  The script parses the `dataset.yaml` (or `data.yaml`) file found in the `--dataset` to get the paths for image and label directories for the specified (or chosen) split, as well as class names .
2.  It then loads images from the image directory and their corresponding `.txt` label files (expected to have the same base name and be in a parallel 'labels' directory, as per YOLO convention) from the label directory for that split .
3.  Each image is displayed in an OpenCV window with its segmentation masks (polygons) and class labels overlaid. If class names are defined in the YAML, they are displayed; otherwise, class IDs are shown .

**Interactive Controls:**

Once an image is displayed, you can use the following commands in the **terminal** where you ran the script:

*   **`n`**: Show the next image.
*   **`p`**: Show the previous image.
*   **`<number>`**: Go to the image at the specified number (1-indexed).
*   **`q`**: Quit the browser.

You can also quit by pressing **`q`** when the **OpenCV image window** is active .

**Example:**

To browse the 'validation' split of a dataset located at `datasets/my_offroad_data`:

```
python3 scripts/browse_dataset.py --dataset_root_dir datasets/my_offroad_data --split val
```

### 3. Train
This script trains Ultralytics segmentation or detection models. It supports configuration via YAML/JSON files, command-line arguments, and offers TensorBoard/MLflow logging. Assumes your script is `scripts/train_model.py`.

**Usage**

Run from the command line:

```
python3 scripts/train_model.py [OPTIONS]
```

**Key Command-Line Arguments**

*   **`--config-file <path>`**: Path to YAML/JSON config file.
*   **`--tensorboard`**: Enable TensorBoard logging.
*   **`--mlflow`**: Enable MLflow logging (sets `MLFLOW_EXPERIMENT_NAME` & `MLFLOW_RUN`).
*   **`--model-name <name>`**: Model to train (e.g., `yolov8l-seg-pt`). Required if not in config.
*   **`--data <path>`**: Path to data configuration file (e.g., `data.yaml`).
*   **`--epochs <number>`**: Override training epochs.
*   **`--batch-size <number>`**: Override batch size (`-1` for auto-batch).
*   **`--imgsz <size>`**: Override input image size.
*   **`--learning-rate <float>`**: Override initial learning rate (lr0).
*   **`--project <name>`**: Project directory for saving results.
*   **`--name <name>`**: Experiment name (subdirectory within project).
*   **`--device <specifier>`**: Device to train on (e.g., 'cpu', '0').
*   **`--workers <number>`**: Number of data loading workers.
*   **`--extra-params '<json_string>'`**: Additional JSON parameters (e.g., `'{"patience": 50}'`). See Ultralytics train settings (<https://docs.ultralytics.com/modes/train/#train-settings>).

**Configuration Precedence**

1.  Direct CLI Arguments (e.g., `--epochs`)
2.  `--extra-params` JSON String
3.  `--config-file`
4.  Ultralytics Defaults

**Examples**

*   **Using a config file:**
    ```
    python3 scripts/train_model.py --config-file configs/my_config.yaml
    ```
    *Example `my_config.yaml`*:
    ```
    model_name: yolov8s-seg.pt
    data: coco128-seg.yaml
    epochs: 50
    batch_size: 16
    imgsz: 640
    project: 'my_project'
    name: 'experiment1'
    ```

*   **CLI overrides:**
    ```
    python3 scripts/train_model.py --config-file configs/my_config.yaml --epochs 100 --name 'longer_run'
    ```

*   **Using only CLI arguments:**
    ```
    python3 scripts/train_model.py --model-name yolov8l-seg-pt --data data.yaml --epochs 75 --project 'new_proj' --name 'run1'
    ```

*   **With `--extra-params` and logging:**
    ```
    python3 scripts/train_model.py \
        --config-file configs/base.yaml \
        --extra-params '{"patience": 20, "optimizer": "AdamW"}' \
        --tensorboard --mlflow
    ```

**Output & Logging**

*   Console logs provide training details.
*   TensorBoard/MLflow logs are generated if enabled.
*   Results are saved in `project/name`.
*   The used config file (if any) is copied to the results directory.

**Model Loading**

The script uses `get_model(model_name)` from `src.models.model_registry` to load the initial Ultralytics model.

### 4. Predict

updating...

### 5. Export

updating...

### 6. Tune Hyperparameters with Optuna

updating...

### 7. Export Best Hyperparameters from Optuna Tuning Results

This script allows you to easily extract the best hyperparameter set found by Optuna from a SQLite database and export it to a YAML file. It is designed for workflows where Optuna is used for hyperparameter optimization and experiment tracking.

#### Features

- **Exports best hyperparameters** from an Optuna study to a YAML file.
- **Optionally updates an existing YAML config**: If you provide an input config, only parameters listed in the `tune` section will be updated or added at the root level.
- **Writes the best objective value** as a comment at the top of the YAML file.

---

#### How to Use

**1. Export all best parameters to a new YAML file:**

```
python export_optuna_best --storage runs/optuna_tunes/results/db.sqlite3 --output best_params.yaml
```

- This creates `best_params.yaml` containing all best hyperparameters and the best value as a comment.

**2. Update an existing config file with best tuned parameters:**

python export_optuna_best --input config.yaml --storage runs/optuna_tunes/results/db.sqlite3 --output best_config.yaml

- This updates only the parameters listed in the `tune` section of `config.yaml` with the best values from Optuna, preserving the rest of your config and its comments.

---

#### Arguments

- `--storage` (required): Path to the Optuna SQLite database file.
- `--input` (optional): Path to your existing YAML config file. If omitted, all best parameters are exported.
- `--output` (optional): Output YAML file name. Defaults to `best_params.yaml`.

---


**Tip:**  
This script is ideal for integrating Optuna’s tuning results directly into your model training or deployment pipelines, ensuring reproducibility and easy experiment management.
