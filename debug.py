import logging
from pathlib import Path

import yaml

from stamp.config import StampConfig
from stamp.encoding import get_slide_embs

# Set up the logger
_logger = logging.getLogger("stamp")
_logger.setLevel(logging.DEBUG)


def run_encode_slides(config_file_path: Path):
    # Load YAML configuration using StampConfig
    with open(config_file_path, "r") as config_yaml:
        config = StampConfig.model_validate(yaml.safe_load(config_yaml))

    # Ensure slide encoding configuration is present
    if config.slide_encoding is None:
        raise ValueError("No slide encoding configuration supplied")

    # Log the configuration being used
    _logger.info(
        "Using the following configuration:\n"
        f"{yaml.dump(config.slide_encoding.model_dump(mode='json'))}"
    )

    # Call the get_slide_embs function with the configuration
    get_slide_embs(
        encoder=config.slide_encoding.encoder,
        output_dir=config.slide_encoding.output_dir,
        feat_dir=config.slide_encoding.feat_dir,
        device=config.slide_encoding.device,
        agg_feat_dir=config.slide_encoding.agg_feat_dir,
    )


# Example usage
if __name__ == "__main__":
    # Path to the config.yaml file
    config_file_path = Path(
        "/mnt/bulk-sirius/juan/pap_screening/stamp_config.yaml"
    )  # Replace with your config file path

    # Run the encode_slides command
    run_encode_slides(config_file_path)
