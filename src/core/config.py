import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration globally
CONFIG = load_config()
