"""
RefLex Configuration File Loading System

This module provides automatic configuration loading from reflex.json files,
similar to how python-dotenv loads .env files. It searches for reflex.json
in the current directory and parent directories, loading configuration
automatically when RefLex components are initialized.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import ValidationError
from pydantic import BaseModel, Field
from reflex_llms.server import ReflexServerConfig


class Config(BaseModel):
    """
    Configuration model for OpenAI provider settings.
    
    Provides configurable endpoints and API versions for different OpenAI
    providers including OpenAI, Azure OpenAI, and RefLex local servers.
    """

    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI API",
    )

    azure_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version",
    )

    azure_base_url: Optional[str] = Field(
        default=None,
        description=
        "Custom Azure OpenAI base URL override. If None, uses AZURE_OPENAI_ENDPOINT env var",
    )

    preference_order: List[str] = Field(
        default=["openai", "azure", "reflex"],
        description="Default provider preference order",
    )

    timeout: float = Field(
        default=5.0,
        ge=0.1,
        description="Default timeout for provider health checks",
    )

    reflex_server: ReflexServerConfig = Field(
        default_factory=ReflexServerConfig,
        description="RefLex server configuration",
    )


def _find_reflex_config(
    start_path: Optional[Union[str, Path]] = None,
    filename: str = "reflex.json",
) -> Optional[Path]:
    """
    Find reflex.json configuration file by searching up the directory tree.
    
    Searches for the configuration file starting from the given path and
    moving up through parent directories until found or reaching the root.
    
    Parameters
    ----------
    start_path : str, Path, or None, default None
        Starting directory for search. If None, uses current working directory
    filename : str, default "reflex.json"
        Name of the configuration file to search for
        
    Returns
    -------
    Path or None
        Path to the configuration file if found, None otherwise
        
    Examples
    --------
    >>> config_path = find_reflex_config()
    >>> if config_path:
    ...     print(f"Found config at: {config_path}")
    """
    if start_path is None:
        start_path = Path.cwd()
    elif isinstance(start_path, str):
        start_path = Path(start_path)

    current_path = start_path.resolve()

    # Search up the directory tree
    while current_path != current_path.parent:
        config_file = current_path / filename
        if config_file.exists() and config_file.is_file():
            return config_file
        current_path = current_path.parent

    # Check root directory
    config_file = current_path / filename
    if config_file.exists() and config_file.is_file():
        return config_file

    return None


def load_reflex_config(
    config_path: Optional[Union[str, Path]] = None,
    search_parents: bool = True,
    filename: str = "reflex.json",
) -> dict:
    """
    Load RefLex configuration from JSON file.
    
    Loads configuration from a JSON file, either from a specified path
    or by searching for the file in the current and parent directories.
    
    Parameters
    ----------
    config_path : str, Path, or None, default None
        Path to configuration file. If None and search_parents is True,
        searches for file automatically
    search_parents : bool, default True
        Whether to search parent directories for configuration file
    filename : str, default "reflex.json"
        Name of configuration file when searching
        
    Returns
    -------
    dict or None
        Configuration dictionary if file found and valid, None otherwise
        
    Raises
    ------
    json.JSONDecodeError
        If configuration file contains invalid JSON
    FileNotFoundError
        If specified config_path doesn't exist
        
    Examples
    --------
    Load from current/parent directories:
    
    >>> config = load_reflex_config()
    >>> if config:
    ...     print(f"Loaded config with keys: {list(config.keys())}")
    
    Load from specific path:
    
    >>> config = load_reflex_config("/path/to/my-config.json")
    """
    # Determine config file path
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    elif search_parents:
        config_file = _find_reflex_config(filename=filename)
        if not config_file:
            raise FileNotFoundError(
                f"Configuration file '{filename}' not found in current or parent directories.")
    else:
        config_file = Path(filename)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file '{filename}' not found in specified path.")

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    print(f"Loaded RefLex configuration from: {config_file}")
    return Config(**config_data).model_dump()


if __name__ == "__main__":
    conf = Config(**load_reflex_config())
    print(conf)
