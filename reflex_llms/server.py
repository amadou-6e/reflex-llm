"""
RefLex OpenAI-Compatible Backend Server Module

This module provides a complete OpenAI-compatible backend server implementation
using Ollama for local model inference. It orchestrates Docker container management,
model setup, and health monitoring to provide a seamless drop-in replacement for
OpenAI's API endpoints.

The ReflexServer class combines container management and model management into a
unified interface that handles the entire lifecycle of an OpenAI-compatible local
AI backend, from initial setup to health monitoring and graceful shutdown.

Key Features
------------
- Complete OpenAI API compatibility layer
- Automated Docker container lifecycle management
- Model pulling, tagging, and OpenAI mapping
- Comprehensive health monitoring and status reporting
- Essential vs. full model setup options
- Automatic restart and recovery capabilities
- Persistent data storage management

Examples
--------
Basic usage with automatic setup:

>>> from reflex_server import ReflexServer
>>> server = ReflexServer()  # Auto-setup enabled by default
>>> print(f"OpenAI endpoint: {server.openai_compatible_url}")
>>> status = server.get_status()

Custom configuration:

>>> server = ReflexServer(
...     port=8080,
...     container_name="my-ai-server",
...     minimal_setup=False,
...     auto_setup=False
... )
>>> success = server.setup()
>>> if success:
...     print("Server ready for requests")

Manual control:

>>> server = ReflexServer(auto_setup=False)
>>> server.setup()
>>> print(f"Health status: {server.is_healthy}")
>>> server.restart()
>>> server.stop()

Dependencies
------------
- time : Time utilities for health checks and timeouts
- pathlib : Path handling for data storage
- typing : Type annotations support
- reflex_llms.containers : Docker container management
- reflex_llms.models : Ollama model management
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# -- Ours --
from reflex_llms.containers import (
    ContainerHandler,
    ContainerConfig,
)
from reflex_llms.models import OllamaManager, OllamaModelManagerConfig
from reflex_llms.settings import *


class ModelMapping(BaseModel):
    """
    Definition class for OpenAI -> Ollama model mappings.
    """
    minimal_setup: bool = Field(
        default=False,
        description="Whether to only setup essential models for faster startup",
    )

    model_mapping: Dict[str, str] = Field(
        default=DEFAULT_MODEL_MAPPINGS,
        description="Mapping of OpenAI model names to Ollama model identifiers.",
    )

    minimal_model_mapping: Dict[str, str] = Field(
        default=DEFAULT_MINIMAL_MODEL_MAPPINGS,
        description=
        "Minimal mapping of OpenAI model names to Ollama model identifiers, with smaller preload times.",
    )


class ReflexServerConfig(ContainerConfig):
    """
    Pydantic configuration model for ReflexServer.
    
    This model provides validation and serialization for ReflexServer
    initialization parameters, ensuring type safety and proper configuration
    management for the OpenAI-compatible backend server.
    """

    auto_setup: bool = Field(
        default=True,
        description="Whether to automatically run setup during initialization",
    )

    model_mappings: ModelMapping = Field(
        default=ModelMapping(),
        description="Model mappings for OpenAI compatibility",
    )


class ReflexServer:
    """
    Complete RefLex OpenAI-compatible backend server.
    
    This class orchestrates the entire lifecycle of an OpenAI-compatible AI backend,
    combining Docker container management, model setup, and health monitoring into
    a unified interface. It provides automatic setup, comprehensive status monitoring,
    and graceful error handling for production deployments.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Host address where Ollama will be accessible
    port : int, default 11434
        Port number for Ollama API access
    container_name : str, default "reflex-server"
        Name for the Docker container
    data_path : Path or None, default None
        Path for persistent data storage. If None, uses default location
    auto_setup : bool, default True
        Whether to automatically run setup during initialization
    minimal_setup : bool, default True
        Whether to only setup essential models for faster startup

    Attributes
    ----------
    host : str
        Host address where Ollama is accessible
    port : int
        Port number for Ollama API access  
    container_name : str
        Name of the Docker container
    data_path : Path or None
        Path for persistent data storage
    minimal_setup : bool
        Whether to use essential models only
    container_handler : ContainerHandler
        Handler for Docker container operations
    model_manager : OllamaModelManager
        Manager for model operations and OpenAI mappings

    Examples
    --------
    Quick start with defaults:

    >>> server = ReflexServer()
    >>> print(f"Server ready at: {server.openai_compatible_url}")

    Custom configuration for production:

    >>> server = ReflexServer(
    ...     host="0.0.0.0",
    ...     port=8080,
    ...     container_name="production-ai-server",
    ...     data_path=Path("/opt/ai-models"),
    ...     minimal_setup=False
    ... )

    Manual setup control:

    >>> server = ReflexServer(auto_setup=False)
    >>> if server.setup():
    ...     print("Setup successful")
    >>> status = server.get_status()
    >>> server.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        image: str = "ollama/ollama:latest",
        container_name: str = "ollama-openai",
        data_path: Optional[Path] = None,
        startup_timeout: int = 120,
        auto_setup: bool = True,
        model_mappings: Dict[str, Any] = {
            "minimal_setup": False,
            "model_mapping": DEFAULT_MODEL_MAPPINGS,
            "minimal_model_mapping": DEFAULT_MINIMAL_MODEL_MAPPINGS
        },
    ) -> None:
        """
        Initialize the RefLex OpenAI-compatible backend server.

        Parameters
        ----------
        host : str, default "127.0.0.1"
            Host address for Ollama API access
        port : int, default 11434
            Port number for Ollama API
        container_name : str, default "reflex-server"
            Name for the Docker container
        data_path : Path or None, default None
            Path for persistent data storage
        auto_setup : bool, default True
            Whether to automatically setup on initialization
        minimal_setup : bool, default True
            Whether to only setup essential models for faster startup
        """
        self.host = host
        self.port = port
        self.container_name = container_name
        self.data_path = data_path
        self.auto_setup = auto_setup
        self.minimal_setup = model_mappings.get("minimal_setup", False)

        # Initialize components
        self.container_handler = ContainerHandler(
            host=host,
            port=port,
            image=image,
            container_name=container_name,
            data_path=data_path,
            startup_timeout=startup_timeout,
        )
        default_selection = model_mappings.get("minimal_model_mapping") \
            if self.minimal_setup else model_mappings.get("model_mapping")

        self.model_manager = OllamaManager(
            ollama_url=self.container_handler.api_url,
            model_mappings=default_selection,
        )

        # Status tracking
        self._setup_complete: bool = False
        self._last_health_check: float = 0
        self._health_check_interval: int = 30  # seconds

        # Auto-setup if requested
        if self.auto_setup:
            self.start()

    def start(self) -> bool:
        """
        Complete setup: container + models + health checks.
        
        Performs the full initialization sequence including container startup,
        model setup, and health verification. This method is idempotent and
        can be safely called multiple times.

        Returns
        -------
        bool
            True if setup completed successfully, False otherwise

        Raises
        ------
        RuntimeError
            If critical setup steps fail (container startup, health check)

        Examples
        --------
        >>> server = ReflexServer(auto_setup=False)
        >>> success = server.setup()
        >>> if success:
        ...     print("Server is ready for requests")

        Notes
        -----
        The setup process includes:
        1. Starting the Ollama Docker container
        2. Waiting for Ollama to become ready
        3. Setting up OpenAI-compatible model mappings
        4. Performing final health verification
        """
        print("Setting up RefLex OpenAI-compatible backend...")

        try:
            # Step 1: Ensure container is running
            print("Starting Ollama container...")
            self.container_handler.ensure_running()

            # Step 2: Wait for Ollama to be fully ready
            print("Waiting for Ollama to be ready...")
            self._wait_for_ollama_ready()

            # Step 3: Set up model mappings
            print("Setting up OpenAI model mappings...")
            success = self.model_manager.setup_openai_models()

            if not success:
                print("Warning: Some models failed to setup, but backend is functional")

            # Step 4: Final health check
            print("Performing final health check...")
            if not self.health_check():
                raise RuntimeError("Health check failed after setup")

            self._setup_complete = True

            print("RefLex OpenAI backend setup complete!")
            print(f"OpenAI-compatible endpoint: {self.openai_compatible_url}")
            print(f"Status endpoint: {self.api_url}/api/tags")

            return True

        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def _wait_for_ollama_ready(self, timeout: int = 120) -> None:
        """
        Wait for Ollama to be fully ready for API calls.
        
        Polls the Ollama API until it responds successfully or timeout is reached.
        This ensures the service is fully initialized before proceeding with
        model operations.

        Parameters
        ----------
        timeout : int, default 120
            Maximum seconds to wait for Ollama readiness

        Raises
        ------
        RuntimeError
            If Ollama doesn't become ready within the timeout period
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to make a simple API call
                models = self.model_manager.list_models()
                print(f"Ollama ready! Found {len(models)} existing models")
                return
            except Exception:
                print("Ollama not ready yet, waiting...")
                time.sleep(3)

        raise RuntimeError(f"Ollama did not become ready within {timeout} seconds")

    def health_check(self, force: bool = False) -> bool:
        """
        Comprehensive health check of the backend.
        
        Performs multiple checks to verify the backend is fully operational,
        including container status, port accessibility, API responsiveness,
        and model availability. Results are cached to avoid excessive checking.

        Parameters
        ----------
        force : bool, default False
            Force health check even if recently performed (ignores cache)

        Returns
        -------
        bool
            True if all health checks pass, False otherwise

        Examples
        --------
        >>> server = ReflexServer()
        >>> if server.health_check():
        ...     print("Server is healthy")
        >>> 
        >>> # Force immediate check
        >>> status = server.health_check(force=True)

        Notes
        -----
        Health checks include:
        1. Docker container running status
        2. Port accessibility verification
        3. API endpoint responsiveness
        4. OpenAI-compatible model availability
        """
        current_time = time.time()

        # Skip if recently checked (unless forced)
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return True

        try:
            # Check 1: Container running
            if not self.container_handler._is_container_running():
                print("Health check failed: Container not running")
                return False

            # Check 2: Port accessible
            if not self.container_handler._is_port_open():
                print("Health check failed: Port not accessible")
                return False

            # Check 3: API responding
            models = self.model_manager.list_models()

            # Check 4: OpenAI-compatible models available
            openai_models = [
                m for m in models
                if any(openai_name in m['name']
                       for openai_name in self.model_manager.model_mappings.keys())
            ]

            self._last_health_check = current_time

            if len(openai_models) == 0:
                print("Health check warning: No OpenAI-compatible models found")
                return True  # Still functional, just no models

            print(
                f"Health check passed: {len(models)} models, {len(openai_models)} OpenAI-compatible"
            )
            return True

        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns detailed information about the backend's current state,
        including setup status, health metrics, model counts, and API endpoints.

        Returns
        -------
        dict
            Dictionary containing comprehensive status information with keys:
            - setup_complete: Whether initial setup finished successfully
            - container_running: Docker container status
            - port_open: API port accessibility
            - total_models: Total number of available models
            - openai_compatible_models: List of OpenAI-compatible model names
            - api_url: Base Ollama API URL
            - openai_compatible_url: OpenAI-compatible API URL
            - healthy: Overall health status
            - error: Error message if status retrieval failed

        Examples
        --------
        >>> server = ReflexServer()
        >>> status = server.get_status()
        >>> print(f"Models available: {len(status['openai_compatible_models'])}")
        >>> print(f"Health status: {status['healthy']}")
        """
        try:
            models = self.model_manager.list_models()
            openai_models = [
                m['name']
                for m in models
                if any(openai_name in m['name']
                       for openai_name in self.model_manager.model_mappings.keys())
            ]

            return {
                "setup_complete": self._setup_complete,
                "container_running": self.container_handler._is_container_running(),
                "port_open": self.container_handler._is_port_open(),
                "total_models": len(models),
                "openai_compatible_models": openai_models,
                "api_url": self.api_url,
                "openai_compatible_url": self.openai_compatible_url,
                "healthy": self.health_check()
            }
        except Exception as e:
            return {"setup_complete": self._setup_complete, "error": str(e), "healthy": False}

    def stop(self) -> None:
        """
        Stop the backend and perform cleanup.
        
        Gracefully shuts down the Docker container and resets internal state.
        This method is safe to call multiple times and handles errors gracefully.

        Examples
        --------
        >>> server = ReflexServer()
        >>> # ... use server ...
        >>> server.stop()  # Clean shutdown
        """
        print("Stopping RefLex OpenAI backend...")
        try:
            self.container_handler.stop()
            self._setup_complete = False
            print("Backend stopped successfully")
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def restart(self) -> bool:
        """
        Restart the backend with fresh initialization.
        
        Performs a complete stop and restart cycle, including full setup.
        This is useful for recovering from errors or applying configuration changes.

        Returns
        -------
        bool
            True if restart completed successfully, False otherwise

        Examples
        --------
        >>> server = ReflexServer()
        >>> if not server.is_healthy:
        ...     success = server.restart()
        ...     print(f"Restart {'successful' if success else 'failed'}")
        """
        print("Restarting RefLex OpenAI backend...")
        self.stop()
        time.sleep(2)
        return self.start()

    @property
    def api_url(self) -> str:
        """
        Get the base Ollama API URL.

        Returns
        -------
        str
            Base URL for direct Ollama API access
        """
        return self.container_handler.api_url

    @property
    def openai_compatible_url(self) -> str:
        """
        Get the OpenAI-compatible API URL.

        Returns
        -------
        str
            URL for OpenAI-compatible API endpoints (includes /v1 suffix)
        """
        return self.container_handler.openai_compatible_url

    @property
    def is_running(self) -> bool:
        """
        Check if backend is currently running.

        Returns
        -------
        bool
            True if the backend is accessible, False otherwise
        """
        return self.container_handler._is_port_open()

    @property
    def is_healthy(self) -> bool:
        """
        Check if backend is healthy and fully operational.

        Returns
        -------
        bool
            True if all health checks pass, False otherwise
        """
        return self.health_check()
