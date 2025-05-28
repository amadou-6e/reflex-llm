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
from typing import Dict, List, Optional, Any, Union
import time
import atexit

from reflex_llms.server import ReflexServer

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


def get_openai_client_config(
    preference_order: Optional[List[str]] = None,
    timeout: float = 5.0,
    force_recheck: bool = False,
) -> Dict[str, Any]:
    """
    Get OpenAI client configuration with intelligent provider resolution.
    
    Attempts to connect to OpenAI-compatible providers in order of preference,
    performing health checks and returning configuration for the first available
    provider. Results are cached for performance optimization.

    Parameters
    ----------
    preference_order : list of str or None, default None
        Provider preference order. If None, defaults to ["openai", "azure", "reflex"]
    timeout : float, default 5.0
        Connection timeout in seconds for provider health checks
    force_recheck : bool, default False
        Force re-checking providers, ignoring cached configuration

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

    Examples
    --------
    Use default provider resolution:

    >>> config = get_openai_client_config()
    >>> print(f"Selected: {config['base_url']}")

    Custom preference with forced recheck:

    >>> config = get_openai_client_config(
    ...     preference_order=["reflex", "openai"],
    ...     force_recheck=True
    ... )

    Notes
    -----
    Provider resolution process:
    1. **OpenAI**: Checks api.openai.com accessibility and OPENAI_API_KEY
    2. **Azure**: Verifies Azure endpoint and required environment variables
    3. **RefLex**: Creates/manages local Ollama server with automatic setup
    
    The first successful provider is cached and reused until force_recheck=True
    or clear_cache() is called.
    """
    global _cached_config, _selected_provider

    # Return cached config if available and not forcing recheck
    if not force_recheck and _cached_config is not None:
        print(f"Using cached {_selected_provider} configuration")
        return _cached_config.copy()

    if preference_order is None:
        preference_order = ["openai", "azure", "reflex"]

    print("Checking OpenAI API providers...")

    for provider in preference_order:
        print(f"  Trying {provider}...")

        if provider == "openai":
            # Check OpenAI API
            try:
                response = requests.get("https://api.openai.com/v1/models",
                                        headers={"Authorization": "Bearer test"},
                                        timeout=timeout)
                if response.status_code in [200, 401]:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        print("✓ Using OpenAI API")
                        config = {"api_key": api_key, "base_url": "https://api.openai.com/v1"}
                        # Cache the result
                        _cached_config = config.copy()
                        _selected_provider = "openai"
                        return config
                    else:
                        print("  OpenAI available but no API key")
            except Exception:
                print("  OpenAI not accessible")

        elif provider == "azure":
            # Check Azure OpenAI
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            if endpoint and api_key:
                try:
                    test_url = f"{endpoint.rstrip('/')}/openai/deployments"
                    response = requests.get(test_url, timeout=timeout)
                    if response.status_code < 500:
                        print("  Using Azure OpenAI")
                        config = {
                            "api_key":
                                api_key,
                            "base_url":
                                test_url,
                            "api_version":
                                os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                        }
                        # Cache the result
                        _cached_config = config.copy()
                        _selected_provider = "azure"
                        return config
                except Exception:
                    pass
            print("  Azure not available or not configured")

        elif provider == "reflex":
            # Use or create RefLex server
            global _reflex_server

            if _reflex_server and _reflex_server.is_healthy:
                print("  Using existing RefLex server")
                config = {"api_key": "reflex", "base_url": _reflex_server.openai_compatible_url}
                _cached_config = config.copy()
                _selected_provider = "reflex"
                return config

            print("  Setting up RefLex server...")
            try:
                # Clean up old server if exists
                if _reflex_server:
                    try:
                        _reflex_server.stop()
                    except Exception:
                        pass

                _reflex_server = ReflexServer(port=11434,
                                              container_name="openai-fallback-server",
                                              auto_setup=True,
                                              essential_models_only=True)

                # Wait for setup (max 5 minutes)
                for i in range(60):
                    if _reflex_server._setup_complete and _reflex_server.is_healthy:
                        print("  Using RefLex local server")
                        config = {
                            "api_key": "reflex",
                            "base_url": _reflex_server.openai_compatible_url
                        }
                        # Cache the result
                        _cached_config = config.copy()
                        _selected_provider = "reflex"
                        return config
                    time.sleep(5)

                # Setup failed
                _reflex_server.stop()
                _reflex_server = None
                print("  RefLex setup failed")
            except Exception as e:
                print(f"  RefLex error: {e}")
                _reflex_server = None

    # Nothing worked
    raise RuntimeError("No OpenAI providers available")

def _hell()-> None:
    


def get_openai_client(preference_order: Optional[List[str]] = None, **kwargs: Any) -> Any:
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
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai package: pip install openai")

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


# Convenience functions
def get_client_dev_mode(**kwargs: Any) -> Any:
    """
    Get client preferring RefLex first for development environments.
    
    Convenience function that prioritizes local RefLex server for development
    work, falling back to cloud providers if local server is unavailable.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments passed to get_openai_client

    Returns
    -------
    openai.OpenAI
        Configured OpenAI client with development-friendly provider priority

    Examples
    --------
    >>> client = get_client_dev_mode()  # Prefers local RefLex
    >>> print(f"Using: {get_selected_provider()}")
    """
    return get_openai_client(["reflex", "openai", "azure"], **kwargs)


def get_client_prod_mode(**kwargs: Any) -> Any:
    """
    Get client preferring cloud APIs first for production environments.
    
    Convenience function that prioritizes cloud providers (OpenAI, Azure)
    for production deployments, using RefLex as a fallback option.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments passed to get_openai_client

    Returns
    -------
    openai.OpenAI
        Configured OpenAI client with production-friendly provider priority

    Examples
    --------
    >>> client = get_client_prod_mode()  # Prefers cloud APIs
    >>> if is_using_reflex():
    ...     print("Fallback to local server")
    """
    return get_openai_client(["openai", "azure", "reflex"], **kwargs)


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
