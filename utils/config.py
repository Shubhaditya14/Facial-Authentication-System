"""Configuration utilities for loading and managing YAML configs.

Provides config loading, saving, and a Config class for dot notation access.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        path: Path where to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class Config:
    """Configuration wrapper allowing dot notation access.

    Wraps a dictionary to allow attribute-style access (e.g., config.model.backbone)
    instead of dictionary-style access (config['model']['backbone']).

    Example:
        >>> config = Config({'model': {'backbone': 'resnet18', 'pretrained': True}})
        >>> config.model.backbone
        'resnet18'
        >>> config.model.pretrained
        True
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Config with a dictionary.

        Args:
            config: Configuration dictionary. If None, creates empty config.
        """
        self._config = config or {}
        self._init_nested()

    def _init_nested(self) -> None:
        """Convert nested dictionaries to Config objects."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                self._config[key] = Config(value)

    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation.

        Args:
            name: Attribute name to access.

        Returns:
            Value associated with the attribute.

        Raises:
            AttributeError: If attribute doesn't exist.
        """
        if name.startswith("_"):
            return super().__getattribute__(name)

        if name in self._config:
            return self._config[name]

        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute value.

        Args:
            name: Attribute name.
            value: Value to set.
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if isinstance(value, dict):
                value = Config(value)
            self._config[name] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config.

        Args:
            key: Key to check.

        Returns:
            True if key exists, False otherwise.
        """
        return key in self._config

    def __repr__(self) -> str:
        """String representation of Config."""
        return f"Config({self._config})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with optional default.

        Args:
            key: Key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            Value for key or default.
        """
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to dictionary.

        Returns:
            Dictionary representation of the config.
        """
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def update(self, other: Dict[str, Any]) -> None:
        """Update config with another dictionary.

        Args:
            other: Dictionary to merge into config.
        """
        for key, value in other.items():
            if isinstance(value, dict):
                if key in self._config and isinstance(self._config[key], Config):
                    self._config[key].update(value)
                else:
                    self._config[key] = Config(value)
            else:
                self._config[key] = value

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load Config from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Config object.
        """
        return cls(load_config(path))

    def save(self, path: Union[str, Path]) -> None:
        """Save Config to a YAML file.

        Args:
            path: Path where to save.
        """
        save_config(self.to_dict(), path)
