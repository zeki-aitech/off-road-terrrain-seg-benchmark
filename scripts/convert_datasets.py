import argparse
import sys
from pathlib import Path
from ultralytics.utils import LOGGER

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_registry import get_converter



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to convert. Options: 'yamaha_seg'",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="The source directory containing the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The output directory to save the converted dataset.",
    )
    args = parser.parse_args()
    
    # Get the converter for the specified dataset
    converter = get_converter(args.dataset, args.source_dir, args.output_dir)
    
    converter.convert()

if __name__ == "__main__":
    main()