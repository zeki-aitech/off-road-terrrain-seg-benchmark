import argparse
import sys
from pathlib import Path
import json
from ultralytics.utils import LOGGER

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset_registry import get_converter


def main():
    parser = argparse.ArgumentParser(description="Dataset Conversion Utility")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to convert. Options should be defined in your registry (e.g., 'yamaha_seg').",
    )
    parser.add_argument(
        "--source-dir",
        type=Path, # Use Path type directly
        required=True,
        help="The source directory containing the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path, # Use Path type directly
        required=True,
        help="The output directory to save the converted dataset.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Additional arguments for the converter in JSON string format (e.g., '{\"key\": \"value\"}').",
    )
    args = parser.parse_args()

    LOGGER.info(f"Starting dataset conversion for '{args.dataset}'")
    LOGGER.info(f"Source directory: {args.source_dir}")
    LOGGER.info(f"Output directory: {args.output_dir}")

    parsed_extra_args = {}
    if args.extra_args:
        try:
            parsed_extra_args = json.loads(args.extra_args)
            if not isinstance(parsed_extra_args, dict):
                LOGGER.error("--extra-args must be a JSON object (dictionary).")
                sys.exit(1)
            LOGGER.info(f"Extra arguments: {parsed_extra_args}")
        except json.JSONDecodeError as e:
            LOGGER.error(f"Invalid JSON string for --extra-args: {e}")
            sys.exit(1)
    
    # Validate source directory
    if not args.source_dir.is_dir():
        LOGGER.error(f"Source directory not found or is not a directory: {args.source_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOGGER.error(f"Could not create output directory {args.output_dir}: {e}")
        sys.exit(1)

    try:
        # Get the converter for the specified dataset
        converter = get_converter(
            args.dataset, 
            str(args.source_dir), # Converters might expect strings, or update them
            str(args.output_dir), # to handle Path objects
            **parsed_extra_args # Pass the parsed dictionary
        )
        
        if converter is None:
            LOGGER.error(f"No converter found for dataset type '{args.dataset}'.")
            sys.exit(1)
            
        LOGGER.info(f"Using converter: {type(converter).__name__}")
        converter.convert()
        LOGGER.info(f"Dataset conversion completed successfully. Output at: {args.output_dir}")

    except Exception as e:
        LOGGER.error(f"An error occurred during conversion: {e}", exc_info=True) # Log traceback
        sys.exit(1)

if __name__ == "__main__":
    main()
