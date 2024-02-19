from pathlib import Path
from typing import (
    Dict,
    Optional,
)

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = ROOT / "config" / "config.yaml"


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config file not found at {CONFIG_FILE_PATH}")


def load_config_file(cfg_path: Optional[Path] = None) -> Optional[Dict]:
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            if not yaml_data:
                raise ValueError("Invalid or empty YAML configuration")
            return yaml_data


cfg = load_config_file()
