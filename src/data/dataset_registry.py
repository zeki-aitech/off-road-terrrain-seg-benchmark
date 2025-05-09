
from .converter import YamahaSegConverter

DATASET_CONVERTERS = {
    "yamaha_seg": YamahaSegConverter,
}

CLASS_MAPPINGS = {
    "yamaha_seg": None,
}

def get_converter(dataset_name, source_dir, output_dir):
    """
    Get the appropriate converter for a dataset
    """
    if dataset_name not in DATASET_CONVERTERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    converter_class = DATASET_CONVERTERS[dataset_name]
    return converter_class(source_dir, output_dir)