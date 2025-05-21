# Off-Road Terrain Segmentation Benchmark

## Tools
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


## Metrics
