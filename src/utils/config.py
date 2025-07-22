import os
import yaml
from loguru import logger
from typing import Any, Dict

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Loads and provides access to application configuration from a YAML file.
    Handles validation, error reporting, and logging.
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            self._validate_config(config)
            return config
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise ConfigError(f"YAML parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise ConfigError(f"Unexpected error loading config: {e}")

    def _validate_config(self, config: Dict[str, Any]):
        required_sections = ["gemini", "qdrant", "sources", "processing", "rag", "ui"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                raise ConfigError(f"Missing required config section: {section}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a top-level config section or value.
        """
        return self._config.get(key, default)

    @property
    def gemini(self) -> Dict[str, Any]:
        return self._config["gemini"]

    @property
    def qdrant(self) -> Dict[str, Any]:
        return self._config["qdrant"]

    @property
    def sources(self) -> Dict[str, Any]:
        return self._config["sources"]

    @property
    def processing(self) -> Dict[str, Any]:
        return self._config["processing"]

    @property
    def rag(self) -> Dict[str, Any]:
        return self._config["rag"]

    @property
    def ui(self) -> Dict[str, Any]:
        return self._config["ui"] 