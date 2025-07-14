"""
Automatically extract information from text via internal decision-making steps of model.
"""

import argparse
import os
import logging
from pathlib import Path
import yaml
from feptp_pipeline.IE.ie_core import Extractor
import weaviate


# Set up logging
def setup_logger(name="ie"):
    config_path = Path(__file__).resolve().parents[2] / "config" / "logging_ie.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f.read())
    log_file = config_path.parent.parent / "logs" / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    config["handlers"]["file"]["filename"] = str(log_file)

    logging.config.dictConfig(config)
    return logging.getLogger(name)


logger = setup_logger("ie")


def validate_path(path, is_dir=False, create_if_missing=False):
    """Validate and convert to absolute path."""
    abs_path = os.path.abspath(path)
    if create_if_missing:
        os.makedirs(abs_path, exist_ok=True)
    if is_dir:
        if not os.path.isdir(abs_path):
            raise FileNotFoundError(f"Directory not found: {abs_path}")
    else:
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
    return abs_path


def initialize_weaviate():
    """Initialize and return Weaviate client."""
    try:
        logger.info("Initializing Weaviate client...")
        return weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Weaviate client: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Document Information Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--doi",
        required=True,
        help="Document DOI to extract"
    )

    # Path configuration
    parser.add_argument(
        "--output-folder",
        default=os.path.join(os.path.dirname(__file__), "../../data/auto_ext"),
        help="Output directory for extracted data"
    )

    # Extraction parameters
    parser.add_argument(
        "--query",
        help="Query parameter for extraction"
    )

    parser.add_argument(
        "--prompt",
        default=os.path.join(os.path.dirname(__file__), "../../prompt/prompt_auto_ext.json"),
        help="Prompt for extraction"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Convert paths to absolute
        output_folder = validate_path(args.output_folder, is_dir=True, create_if_missing=True)

        # Initialize clients
        wv_client = initialize_weaviate()

        # Initialize extractor
        extractor = Extractor(
            wv_client=wv_client,
            output_folder=output_folder
        )

        logger.info(f"Starting extraction for DOI: {args.doi}")
        response = extractor.extract(
            doi=args.doi,
            query=args.query,
            prompt=args.prompt,
        )
        logger.info(f"{args.doi} extraction response: {response}")
        logger.info("Extraction completed successfully")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
