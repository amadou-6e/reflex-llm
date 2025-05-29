"""
OpenAI API Provider Resolution and Routing Module

This module provides intelligent OpenAI API provider resolution with automatic
fallback capabilities. It manages multiple OpenAI-compatible providers (OpenAI,
Azure OpenAI, and local RefLex servers) with caching, health checking, and
graceful fallback mechanisms.

The module maintains global state to cache provider configurations and manage
RefLex server instances, providing seamless switching between cloud and local
AI providers based on availability and preference.

Key Features
------------
- Multi-provider support (OpenAI, Azure OpenAI, RefLex local)
- Intelligent provider resolution with configurable preference orders
- Configuration caching for performance optimization
- Automatic RefLex server setup and management
- Health checking and failover capabilities
- Development vs production mode convenience functions
- Comprehensive status reporting and monitoring
- Graceful cleanup and resource management

Provider Types
--------------
- **OpenAI**: Official OpenAI API (requires OPENAI_API_KEY)
- **Azure**: Azure OpenAI Service (requires AZURE_OPENAI_* environment variables) 
- **RefLex**: Local Ollama-based server (automatically managed)

Examples
--------
Basic usage with automatic provider resolution:

>>> from openai_routing import get_openai_client
>>> client = get_openai_client()  # Uses cached config after first call
>>> response = client.chat.completions.create(
...     model="gpt-3.5-turbo",
...     messages=[{"role": "user", "content": "Hello!"}]
... )

Development mode (prefers local RefLex):

>>> client = get_client_dev_mode()
>>> print(f"Using provider: {get_selected_provider()}")

Production mode (prefers cloud APIs):

>>> client = get_client_prod_mode()
>>> if is_using_reflex():
...     print("Fallback to local server")

Custom provider preference:

>>> client = get_openai_client(["azure", "reflex", "openai"])
>>> status = get_module_status()

Environment Variables
--------------------
- OPENAI_API_KEY: OpenAI API authentication key
- AZURE_OPENAI_ENDPOINT: Azure OpenAI service endpoint URL
- AZURE_OPENAI_API_KEY: Azure OpenAI authentication key
- AZURE_OPENAI_API_VERSION: Azure API version (optional, defaults to 2024-02-15-preview)

Dependencies
------------
- os : Environment variable access
- requests : HTTP client for provider health checks
- typing : Type annotations support
- time : Timing utilities for setup polling
- atexit : Cleanup registration
- openai : OpenAI Python client (optional, imported when needed)
- reflex_llms.server : Local RefLex server management
"""

import os
import requests
import time
import openai
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from reflex_llms.server import (
    ReflexServer,
    ReflexServerConfig,
    ModelMapping,
)

from reflex_llms.configs import load_reflex_config, Config

# Module-level state
_cached_config: Optional[Dict[str, Any]] = None
_reflex_server: Optional[ReflexServer] = None
_selected_provider: Optional[str] = None


def _cleanup_module_state() -> None:
    """
    Clean up module state on exit.
    
    Automatically called when the module is unloaded to ensure proper
    cleanup of RefLex server resources and prevent resource leaks.
    """
    global _reflex_server
    if _reflex_server:
        try:
            _reflex_server.stop()
        except Exception:
            pass
        _reflex_server = None


# Register cleanup on module exit
atexit.register(_cleanup_module_state)


def get_openai_client_type(
    preference_order: Optional[List[str]] = None,
    timeout: Optional[float] = 120.0,
    force_recheck: bool = False,
    openai_base_url: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_base_url: Optional[str] = None,
    reflex_server_config: Optional[ReflexServerConfig] = None,
    from_file: bool = False,
    **kwargs: Any,
) -> str:
    """
    Get OpenAI client configuration type with intelligent provider resolution.
    
    Attempts to connect to OpenAI-compatible providers in order of preference,
    performing health checks and returning the provider type string for the first 
    available provider. Optionally loads configuration from reflex.json file.

    Parameters
    ----------
    preference_order : list of str or None, default None
        Provider preference order. If None, defaults to ["openai", "azure", "reflex"]
    timeout : float or None, default None
        Connection timeout in seconds for provider health checks. If None, defaults to 5.0
    force_recheck : bool, default False
        Force re-checking providers, ignoring cached configuration
    openai_base_url : str or None, default None
        Base URL for OpenAI API. If None, defaults to "https://api.openai.com/v1"
    azure_api_version : str or None, default None
        Azure OpenAI API version. If None, defaults to "2024-02-15-preview"
    azure_base_url : str or None, default None
        Custom Azure OpenAI base URL override. If None, uses AZURE_OPENAI_ENDPOINT env var
    reflex_server_config : ReflexServerConfig or None, default None
        RefLex server configuration object. If None, uses default configuration
    from_file : bool, default False
        Whether to automatically discover and load configuration from reflex.json
    **kwargs : Any
        Additional keyword arguments for RefLex server configuration

    Returns
    -------
    str
        Provider type string: "openai", "azure", or "reflex"

    Raises
    ------
    RuntimeError
        If no providers are available or accessible
    """
    global _cached_config, _selected_provider

    # Resolve configuration parameters using encapsulated function
    config = _resolve_configuration_parameters(
        from_file=from_file,
        preference_order=preference_order,
        timeout=timeout,
        openai_base_url=openai_base_url,
        azure_api_version=azure_api_version,
        azure_base_url=azure_base_url,
        reflex_server=reflex_server_config,
    )

    # Extract resolved values
    preference_order = config['preference_order']
    timeout = config['timeout']
    openai_base_url = config['openai_base_url']
    azure_api_version = config['azure_api_version']
    azure_base_url = config['azure_base_url']
    reflex_server_config = config['reflex_server']

    # Return cached provider type if available and not forcing recheck
    if not force_recheck and _selected_provider is not None:
        print(f"Using cached {_selected_provider} configuration")
        return _selected_provider

    print("Checking OpenAI API providers...")

    for provider in preference_order:
        print(f"  Trying {provider}...")

        if provider == "openai":
            client_config = _try_openai_provider(
                timeout=timeout,
                base_url=openai_base_url,
            )
            if client_config:
                _cached_config = client_config.copy()
                _selected_provider = "openai"
                return "openai"

        elif provider == "azure":
            client_config = _try_azure_provider(
                timeout=timeout,
                api_version=azure_api_version,
                base_url=azure_base_url,
            )
            if client_config:
                _cached_config = client_config.copy()
                _selected_provider = "azure"
                return "azure"

        elif provider == "reflex":
            client_config = _try_reflex_provider(
                reflex_server_config=reflex_server_config,
                **kwargs,
            )
            if client_config:
                _cached_config = client_config.copy()
                _selected_provider = "reflex"
                return "reflex"

    # Nothing worked
    raise RuntimeError("No OpenAI providers available")


def get_openai_client_config(
    preference_order: Optional[List[str]] = None,
    timeout: Optional[float] = None,
    force_recheck: bool = False,
    openai_base_url: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_base_url: Optional[str] = None,
    reflex_server_config: Optional[ReflexServerConfig] = None,
    from_file: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Get OpenAI client configuration dictionary with intelligent provider resolution.
    
    This function returns the actual configuration dictionary needed to initialize
    an OpenAI client, while get_openai_client_config returns only the provider type.

    Parameters
    ----------
    preference_order : list of str or None, default None
        Provider preference order. If None, defaults to ["openai", "azure", "reflex"]
    timeout : float or None, default None
        Connection timeout in seconds for provider health checks. If None, defaults to 5.0
    force_recheck : bool, default False
        Force re-checking providers, ignoring cached configuration
    openai_base_url : str or None, default None
        Base URL for OpenAI API. If None, defaults to "https://api.openai.com/v1"
    azure_api_version : str or None, default None
        Azure OpenAI API version. If None, defaults to "2024-02-15-preview"
    azure_base_url : str or None, default None
        Custom Azure OpenAI base URL override. If None, uses AZURE_OPENAI_ENDPOINT env var
    reflex_server_config : ReflexServerConfig or None, default None
        RefLex server configuration object. If None, uses default configuration
    from_file : bool, default False
        Whether to automatically discover and load configuration from reflex.json
    **kwargs : Any
        Additional keyword arguments for RefLex server configuration

    Returns
    -------
    dict
        Dictionary containing OpenAI client configuration with keys:
        - api_key: Authentication key for the selected provider
        - base_url: Base URL for API endpoints
        - api_version: API version (Azure only)

    Raises
    ------
    RuntimeError
        If no providers are available or accessible
    """
    global _cached_config, _selected_provider

    # Resolve configuration parameters using encapsulated function
    config = _resolve_configuration_parameters(
        from_file=from_file,
        preference_order=preference_order,
        timeout=timeout,
        openai_base_url=openai_base_url,
        azure_api_version=azure_api_version,
        azure_base_url=azure_base_url,
        reflex_server=reflex_server_config,
    )

    # Extract resolved values
    preference_order = config['preference_order']
    timeout = config['timeout']
    openai_base_url = config['openai_base_url']
    azure_api_version = config['azure_api_version']
    azure_base_url = config['azure_base_url']
    reflex_server_config = config['reflex_server']

    # Return cached config if available and not forcing recheck
    if not force_recheck and _cached_config is not None:
        print(f"Using cached {_selected_provider} configuration")
        return _cached_config.copy()

    print("Checking OpenAI API providers...")

    for provider in preference_order:
        print(f"  Trying {provider}...")

        if provider == "openai":
            client_config = _try_openai_provider(
                timeout=timeout,
                base_url=openai_base_url,
            )
            if client_config:
                _cached_config = client_config.copy()
                _selected_provider = "openai"
                return client_config

        elif provider == "azure":
            client_config = _try_azure_provider(
                timeout=timeout,
                api_version=azure_api_version,
                base_url=azure_base_url,
            )
            if client_config:
                _cached_config = client_config.copy()
                _selected_provider = "azure"
                return client_config

        elif provider == "reflex":
            client_config = _try_reflex_provider(
                reflex_server_config=reflex_server_config,
                timeout=timeout,
                **kwargs,
            )
            if client_config:
                _cached_config = client_config.copy()
                _selected_provider = "reflex"
                return client_config

    # Nothing worked
    raise RuntimeError("No OpenAI providers available")


def _resolve_configuration_parameters(
    from_file: bool = False,
    config_path: Optional[Union[str, Path]] = None,
    filename: str = "reflex.json",
    **kwargs,
) -> Config:
    """Resolve configuration with clean dict merging."""

    config_data = {}

    # Load from file first (lowest priority)
    if from_file:
        try:
            config_data = load_reflex_config(config_path=config_path, filename=filename)
            print("Loaded configuration from reflex.json")
        except Exception as e:
            print(f"Failed to load configuration: {e}")

    # Override with provided parameters (highest priority)
    config_data.update({k: v for k, v in kwargs.items() if v is not None})

    # Validate then dump
    return Config(**config_data).model_dump()


def _try_openai_provider(timeout: float, base_url: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to configure OpenAI provider with configurable settings.
    
    Parameters
    ----------
    timeout : float
        Connection timeout in seconds
    base_url : str
        Base URL for OpenAI API
        
    Returns
    -------
    dict or None
        OpenAI configuration if successful, None otherwise
    """
    try:
        # Use configurable base URL for health check
        models_url = f"{base_url.rstrip('/')}/models"
        response = requests.get(models_url,
                                headers={"Authorization": "Bearer test"},
                                timeout=timeout)
        if response.status_code in [200, 401]:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("  Using OpenAI API")
                return {"api_key": api_key, "base_url": base_url}
            else:
                print("  OpenAI available but no API key")
    except Exception:
        print("  OpenAI not accessible")

    return None


def _try_azure_provider(
    timeout: float,
    api_version: str,
    base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Attempt to configure Azure OpenAI provider with configurable settings.
    
    Parameters
    ----------
    timeout : float
        Connection timeout in seconds
    api_version : str
        Azure OpenAI API version
    base_url : str or None, default None
        Custom Azure base URL. If None, uses AZURE_OPENAI_ENDPOINT env var
        
    Returns
    -------
    dict or None
        Azure OpenAI configuration if successful, None otherwise
    """
    # Use provided base URL or fall back to environment variable
    endpoint = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if endpoint and api_key:
        try:
            test_url = f"{endpoint.rstrip('/')}/openai/deployments"
            response = requests.get(test_url, timeout=timeout)
            if response.status_code < 500:
                print("  Using Azure OpenAI")
                return {"api_key": api_key, "base_url": test_url, "api_version": api_version}
        except Exception:
            pass

    print("  Azure not available or not configured")
    return None


def _try_reflex_provider(
    reflex_server_config: Optional[ReflexServerConfig] = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Attempt to configure RefLex local provider with configurable settings.
    
    Parameters
    ----------
    reflex_server_config : ReflexServerConfig or None, default None
        RefLex server configuration object
    **kwargs : Any
        Additional configuration parameters for RefLex server
    
    Returns
    -------
    dict or None
        RefLex configuration if successful, None otherwise
    """
    global _reflex_server

    if isinstance(reflex_server_config, dict):
        reflex_server_config = ReflexServerConfig(**reflex_server_config)

    if _reflex_server and _reflex_server.is_healthy:
        print("  Using existing RefLex server")
        return {"api_key": "reflex", "base_url": _reflex_server.openai_compatible_url}

    print("  Setting up RefLex server...")
    try:
        # Clean up old server if exists
        if _reflex_server:
            try:
                _reflex_server.stop()
            except Exception:
                pass

        # Create RefLex server with provided configuration or defaults
        if reflex_server_config:
            _reflex_server = ReflexServer(**reflex_server_config.model_dump())
        else:
            # Create default configuration with any provided kwargs
            default_config = ReflexServerConfig(
                port=kwargs.get('port', 11434),
                container_name=kwargs.get('container_name', "openai-fallback-server"),
                auto_setup=kwargs.get('auto_setup', True),
                model_mappings=kwargs.get('model_mappings', ModelMapping(minimal_setup=True)),
                host=kwargs.get('host', '127.0.0.1'),
                image=kwargs.get('image', 'ollama/ollama:latest'),
                data_path=kwargs.get('data_path'),
                startup_timeout=kwargs.get('startup_timeout', 120))
            _reflex_server = ReflexServer(**default_config.model_dump())

        if not _reflex_server.auto_setup:
            _reflex_server.start()

        # Wait for setup (max 5 minutes)
        for i in range(max(int(timeout / 5), 1)):
            if _reflex_server._setup_complete and _reflex_server.is_healthy:
                print("  Using RefLex local server")
                return {"api_key": "reflex", "base_url": _reflex_server.openai_compatible_url}
            time.sleep(5)

        # Setup failed
        _reflex_server.stop()
        _reflex_server = None
        print("  RefLex setup failed")
    except Exception as e:
        print(f"  RefLex error: {e}")
        _reflex_server = None

    return None


def get_openai_client(preference_order: Optional[List[str]] = None, **kwargs: Any) -> openai.OpenAI:
    """
    Get configured OpenAI client using cached configuration.
    
    Creates an OpenAI client instance using the resolved provider configuration.
    This is the primary interface for obtaining OpenAI clients with automatic
    provider resolution and caching.

    Parameters
    ----------
    preference_order : list of str or None, default None
        Provider preference order passed to get_openai_client_config
    **kwargs : Any
        Additional keyword arguments passed to get_openai_client_config

    Returns
    -------
    openai.OpenAI
        Configured OpenAI client instance

    Raises
    ------
    ImportError
        If the openai package is not installed
    RuntimeError
        If no providers are available

    Examples
    --------
    Basic usage:

    >>> client = get_openai_client()
    >>> response = client.chat.completions.create(
    ...     model="gpt-3.5-turbo",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )

    With custom preferences:

    >>> client = get_openai_client(["reflex", "openai"])
    """
    config = get_openai_client_config(preference_order, **kwargs)
    return openai.OpenAI(**config)


def get_reflex_server() -> Optional[ReflexServer]:
    """
    Get the RefLex server instance if it was created during resolution.
    
    Provides access to the managed RefLex server instance for direct
    interaction, monitoring, or advanced configuration.

    Returns
    -------
    ReflexServer or None
        RefLex server instance if currently using RefLex provider,
        None otherwise

    Examples
    --------
    >>> client = get_openai_client()
    >>> server = get_reflex_server()
    >>> if server:
    ...     status = server.get_status()
    ...     print(f"Server health: {server.is_healthy}")
    """
    global _reflex_server, _selected_provider

    if _selected_provider == "reflex" and _reflex_server:
        return _reflex_server

    return None


def get_selected_provider() -> Optional[str]:
    """
    Get the currently selected provider name.

    Returns
    -------
    str or None
        Provider name ("openai", "azure", "reflex") or None if not resolved

    Examples
    --------
    >>> client = get_openai_client()
    >>> provider = get_selected_provider()
    >>> print(f"Using provider: {provider}")
    """
    return _selected_provider


def is_using_reflex() -> bool:
    """
    Check if currently using RefLex local server.

    Returns
    -------
    bool
        True if using RefLex provider, False otherwise

    Examples
    --------
    >>> client = get_openai_client()
    >>> if is_using_reflex():
    ...     print("Using local AI server")
    ... else:
    ...     print("Using cloud AI service")
    """
    return _selected_provider == "reflex"


def clear_cache() -> None:
    """
    Clear cached configuration and force re-resolution on next call.
    
    Resets the module state to force provider re-resolution on the next
    call to get_openai_client_config. Useful when network conditions
    change or when switching between environments.

    Examples
    --------
    >>> clear_cache()  # Force provider re-check
    >>> client = get_openai_client()  # Will re-resolve providers
    """
    global _cached_config, _selected_provider
    _cached_config = None
    _selected_provider = None
    print("Cleared provider cache")


def stop_reflex_server() -> None:
    """
    Stop the RefLex server if running.
    
    Gracefully shuts down the managed RefLex server and cleans up resources.
    Safe to call even if no RefLex server is running.

    Examples
    --------
    >>> stop_reflex_server()  # Clean shutdown
    >>> clear_cache()  # Force re-resolution
    """
    global _reflex_server
    if _reflex_server:
        try:
            _reflex_server.stop()
            print("Stopped RefLex server")
        except Exception as e:
            print(f"Error stopping RefLex server: {e}")
        finally:
            _reflex_server = None


def get_module_status() -> Dict[str, Any]:
    """
    Get current module state information.
    
    Provides comprehensive information about the current provider resolution
    state, caching status, and RefLex server health.

    Returns
    -------
    dict
        Dictionary containing module status with keys:
        - selected_provider: Currently selected provider name
        - has_cached_config: Whether configuration is cached
        - reflex_server_running: RefLex server health status
        - reflex_server_url: RefLex server URL if available

    Examples
    --------
    >>> status = get_module_status()
    >>> print(f"Provider: {status['selected_provider']}")
    >>> print(f"Cached: {status['has_cached_config']}")
    >>> if status['reflex_server_running']:
    ...     print(f"RefLex URL: {status['reflex_server_url']}")
    """
    return {
        "selected_provider":
            _selected_provider,
        "has_cached_config":
            _cached_config is not None,
        "reflex_server_running":
            _reflex_server is not None and _reflex_server.is_healthy if _reflex_server else False,
        "reflex_server_url":
            _reflex_server.openai_compatible_url if _reflex_server else None
    }


# Example usage
if __name__ == "__main__":
    print("=== First Resolution ===")
    client1 = get_openai_client()
    print(f"Status: {get_module_status()}")

    print("\n=== Second Call (should use cache) ===")
    client2 = get_openai_client()

    print("\n=== Get RefLex Server ===")
    server = get_reflex_server()
    if server:
        print(f"RefLex server available: {server.openai_compatible_url}")
        print(f"Server healthy: {server.is_healthy}")
    else:
        print("Not using RefLex server")

    print(f"\nSelected provider: {get_selected_provider()}")
    print(f"Using RefLex: {is_using_reflex()}")
