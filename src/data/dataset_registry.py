
from .converter import YamahaSegConverter

DATASET_CONVERTERS = {
    "yamaha_seg": YamahaSegConverter,
}

def get_converter(dataset_name, source_dir, output_dir, **kwargs):
    """
    Get the appropriate converter for a dataset
    """
    if dataset_name not in DATASET_CONVERTERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    converter_class = DATASET_CONVERTERS[dataset_name]
    for arg in kwargs:
        if arg not in converter_class.__init__.__code__.co_varnames:
            raise ValueError(f"Unknown argument '{arg}' for {dataset_name} converter.")
        
    return converter_class(
        source_dir=source_dir, 
        output_dir=output_dir,
        **kwargs
    )