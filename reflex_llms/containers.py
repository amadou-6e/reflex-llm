"""
Ollama Docker Container Management Module

This module provides a comprehensive handler for managing Ollama Docker containers,
offering automated container lifecycle management, health checking, and API connectivity.

The ContainerHandler class abstracts away the complexity of Docker operations and provides
a simple interface for ensuring Ollama is running and accessible via its OpenAI-compatible
API endpoints.

Key Features
------------
- Automatic Docker container creation and management
- Health checking and readiness verification
- Persistent data storage configuration
- OpenAI-compatible API endpoint exposure
- Graceful error handling and recovery

Examples
--------
Basic usage with default settings:

>>> from container_handler import ContainerHandler
>>> handler = ContainerHandler()
>>> handler.ensure_running()
>>> print(f"Ollama API available at: {handler.api_url}")

Custom configuration:

>>> handler = ContainerHandler(
...     port=8080,
...     container_name="my-ollama",
...     data_path=Path("/custom/data/path"),
...     startup_timeout=180
... )
>>> handler.ensure_running()

Dependencies
------------
- docker : Docker SDK for Python
- requests : HTTP library for health checks
- pathlib : Path handling utilities
"""

import docker
import requests
import time
import uuid
import socket
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
# -- Ours --
from reflex_llms.settings import *


class ContainerConfig(BaseModel):
    """Configuration interface for the ContainerManager.
    """
    host: str = Field(
        default="127.0.0.1",
        description="Host address for the container",
    )
    port: int = Field(
        default=11434,
        description="Port number for the container",
    )
    image: str = Field(
        default="ollama/ollama:latest",
        description="Docker image to use",
    )
    container_name: str | None = Field(
        default=None,
        description=
        "Name of the container. Will derive randomized name from uuid generation if not provided",
    )
    data_path: Optional[Path] = Field(
        default=None,
        description="Path for data storage",
    )
    startup_timeout: int = Field(
        default=120,
        description="Timeout in seconds for container startup",
    )

    def model_post_init(self, context):
        if self.container_name == None:
            # Generate a unique container name if not provided
            self.container_name = f"ollama-reflex-{uuid.uuid4().hex[:8]}"
        return


class ContainerHandler:
    """
    Handles Ollama Docker container operations with automated lifecycle management.
    
    This class provides a high-level interface for managing Ollama Docker containers,
    including creation, startup, health monitoring, and API connectivity. It handles
    persistent data storage and ensures the container is properly configured for
    OpenAI-compatible API access.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        The host address where Ollama will be accessible
    port : int, default 11434
        The port number for Ollama API access
    image : str, default "ollama/ollama:latest"
        Docker image name and tag to use
    container_name : str, default "ollama-openai-backend"
        Unique name for the Docker container
    data_path : Path or None, default None
        Local path for persistent Ollama data storage. If None, uses ~/.ollama-docker
    startup_timeout : int, default 120
        Maximum seconds to wait for container startup

    Attributes
    ----------
    host : str
        The host address where Ollama will be accessible
    port : int
        The port number for Ollama API access
    image : str
        Docker image name and tag to use
    container_name : str
        Unique name for the Docker container
    data_path : Path
        Local path for persistent Ollama data storage
    startup_timeout : int
        Maximum seconds to wait for container startup
    client : docker.DockerClient or None
        Docker client instance for container operations

    Examples
    --------
    Basic usage with default settings:

    >>> handler = ContainerHandler()
    >>> handler.ensure_running()
    >>> print(f"Ollama API available at: {handler.api_url}")

    Custom configuration:

    >>> handler = ContainerHandler(
    ...     port=8080,
    ...     container_name="my-ollama",
    ...     data_path=Path("/custom/data/path"),
    ...     startup_timeout=180
    ... )
    >>> handler.ensure_running()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        image: str = "ollama/ollama:latest",
        container_name: str = "ollama-openai",
        data_path: Optional[Path] = None,
        startup_timeout: int = 120,
    ):
        """
        Initialize the Ollama container handler with configuration parameters.

        Parameters
        ----------
        host : str, default "127.0.0.1"
            Host address for Ollama API access
        port : int, default 11434
            Port number for Ollama API
        image : str, default "ollama/ollama:latest"
            Docker image to use for the container
        container_name : str, default "ollama-openai-backend"
            Name for the Docker container
        data_path : Path or None, default None
            Directory for persistent Ollama data. If None, uses ~/.ollama-docker
        startup_timeout : int, default 120
            Maximum seconds to wait for startup

        Raises
        ------
        OSError
            If data directory cannot be created
        docker.errors.DockerException
            If Docker client initialization fails
        """
        self.host = host
        self.port = port
        self.image = image
        self.container_name = container_name
        self.startup_timeout = startup_timeout

        # Set up data path for persistent storage
        self.data_path = data_path or Path.home() / ".ollama-docker"
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.client = None
        if self._is_docker_running():
            self.client = docker.from_env()

    def _is_docker_running(self) -> bool:
        """
        Check if Docker daemon is running.

        Returns
        -------
        bool
            True if Docker daemon is accessible, False otherwise
        """
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def _is_port_open(self) -> bool:
        """
        Check if Ollama port is responding to API requests.

        Returns
        -------
        bool
            True if Ollama API is responding, False otherwise
        """
        try:
            response = requests.get(f"http://{self.host}:{self.port}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _get_container(self):
        """
        Get container instance by name.

        Returns
        -------
        docker.models.containers.Container or None
            Container instance if found, None otherwise
        """
        if not self.client:
            return None
        try:
            return self.client.containers.get(self.container_name)
        except docker.errors.NotFound:
            return None

    def _is_container_running(self) -> bool:
        """
        Check if container exists and is in running state.

        Returns
        -------
        bool
            True if container exists and is running, False otherwise
        """
        container = self._get_container()
        if not container:
            return False
        container.reload()
        return container.status == "running"

    def _pull_image(self):
        """
        Pull Ollama Docker image if not present locally.

        Raises
        ------
        RuntimeError
            If Docker client is not available
        docker.errors.ImageNotFound
            If the specified image cannot be found
        docker.errors.APIError
            If there's an error pulling the image
        """
        if not self.client:
            raise RuntimeError("Docker client not available")

        try:
            self.client.images.get(self.image)
        except docker.errors.NotFound:
            print(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)

    def _create_container(self):
        """
        Create Ollama container with proper configuration.

        Creates a new container with volume mounts for persistent data storage
        and port mapping for API access. Removes any existing container with
        the same name.

        Returns
        -------
        docker.models.containers.Container
            The created container instance

        Raises
        ------
        RuntimeError
            If Docker client is not available
        docker.errors.APIError
            If container creation fails
        """
        if not self.client:
            raise RuntimeError("Docker client not available")

        # Remove existing container if it exists
        existing = self._get_container()
        if existing:
            existing.remove(force=True)

        mounts = [
            docker.types.Mount(
                target="/root/.ollama",
                source=str(self.data_path.resolve()),
                type="bind",
            )
        ]

        ports = {f"11434/tcp": self.port}

        container = self.client.containers.create(
            image=self.image,
            name=self.container_name,
            mounts=mounts,
            ports=ports,
            detach=True,
        )
        return container

    def _start_container(self):
        """
        Start the Ollama container.

        Raises
        ------
        RuntimeError
            If container is not found
        docker.errors.APIError
            If container start fails
        """
        container = self._get_container()
        if not container:
            raise RuntimeError("Container not found")

        container.start()
        print(f"Started Ollama container: {self.container_name}")

    def _wait_for_ready(self, timeout: int = 120):
        """
        Wait for Ollama service to be ready and responding.

        Parameters
        ----------
        timeout : int, default 120
            Maximum seconds to wait for service readiness

        Raises
        ------
        ConnectionError
            If service doesn't become ready within timeout period
        """
        print("Waiting for Ollama to be ready...")
        start_time = time.time()

        while (timeout is None) or (time.time() - start_time < timeout):
            if self._is_port_open():
                print("Ollama is ready")
                return True
            time.sleep(3)

        raise ConnectionError(f"Ollama did not become ready within {timeout} seconds")

    @staticmethod
    def clear_port(
        port: int,
        container_prefix: str = "",
        port_inside_container: int = 11434,
    ):
        """
        Clear a port by stopping Docker containers that are using it.
        
        Continuously monitors and stops containers using the specified port until
        the port is no longer in use. Only stops containers whose names start
        with the given prefix.
        
        Parameters
        ----------
        port : int
            Port number to clear of Docker containers.
        container_prefix : str, optional
            Prefix to match container names. Only containers with names starting
            with this prefix will be stopped. Default is empty string (matches all).
            
        Notes
        -----
        This method will loop until no Docker containers are using the specified port.
        It includes a 2-second wait after clearing to ensure port is fully released.
        """
        client = docker.from_env()

        while True:
            containers = client.containers.list(all=True)  # include stopped/exited containers
            port_still_in_use = False

            for container in containers:
                container.reload()
                ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
                inside = f"{port_inside_container}/tcp"
                if inside in ports:
                    for mapping in ports[inside]:
                        if mapping.get("HostPort", None) is not None \
                        and mapping.get("HostPort") == str(port):
                            port_still_in_use = True
                            if container.name.startswith(container_prefix):
                                container.stop()

            if not port_still_in_use:
                break

            time.sleep(0.5)

        time.sleep(2)

    @staticmethod
    def is_port_in_use(port: int, host: str = "localhost") -> bool:
        """
        Check if a port is in use by network connectivity and Docker containers.
        
        Performs a two-stage check: first attempts network connection to the port,
        then examines Docker containers for port bindings.
        
        Parameters
        ----------
        port : int
            Port number to check for usage.
        host : str, optional
            Hostname or IP address to check. Default is "localhost".
            
        Returns
        -------
        bool
            True if port is in use by either network service or Docker container,
            False if port appears to be available.
            
        Notes
        -----
        Network check uses a 1-second timeout. Docker API errors are silently
        ignored and treated as "port not in use".
        """
        # First check if something is responding on the port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                if result == 0:  # Connection successful = port in use
                    return True
        except Exception:
            pass

        # Then check Docker containers
        try:
            client = docker.from_env()
            containers = client.containers.list(all=True)

            for container in containers:
                container.reload()
                ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
                if f"{port}/tcp" in ports and ports[f"{port}/tcp"] is not None:
                    return True

        except Exception:
            pass

        return False

    def delete(self, force: bool = False, remove_volumes: bool = False) -> bool:
        """
        Delete the container and optionally its associated volumes.
        
        Parameters
        ----------
        force : bool, optional
            If True, force removal even if container is running by stopping it first.
            If False, raises RuntimeError for running containers. Default is False.
        remove_volumes : bool, optional
            If True, remove volumes associated with the container. Default is False.
            
        Returns
        -------
        bool
            True if container was successfully deleted, False if container did not exist.
            
        Raises
        ------
        RuntimeError
            If Docker is not running, or if container is running and force=False.
        docker.errors.APIError
            If Docker API operations fail during container deletion.
            
        Notes
        -----
        Running containers are stopped with a 10-second timeout when force=True.
        """
        if not self._is_docker_running():
            raise RuntimeError("Docker is not running")

        container = self._get_container()
        if not container:
            print(f"Container {self.container_name} does not exist")
            return False

        try:
            # Stop container first if it's running and force is True
            container.reload()
            if container.status == "running":
                if force:
                    print(f"Stopping running container {self.container_name}...")
                    container.stop(timeout=10)
                else:
                    raise RuntimeError(
                        f"Container {self.container_name} is running. Use force=True to stop and remove it."
                    )

            # Remove the container
            print(f"Removing container {self.container_name}...")
            container.remove(v=remove_volumes, force=force)

            print(f"Successfully deleted container {self.container_name}")
            return True

        except Exception as e:
            print(f"Failed to delete container {self.container_name}: {e}")
            raise

    def start(
        self,
        exists_ok: bool = True,
        force: bool = False,
        restart: bool = False,
        attach_port: bool = True,
    ) -> bool:
        """
        Ensure Ollama container is running and ready to accept requests.

        This method handles all aspects of container lifecycle management:
        - Checks if service is already running
        - Verifies Docker availability  
        - Starts existing containers or creates new ones as needed
        - Waits for service readiness before returning

        Returns True, if a new container was created or an existing one was restarted.
        Returns False, if the service was already running and no action was taken.

        Raises
        ------
        RuntimeError
            If Docker is not running or available
        ConnectionError
            If container fails to become ready within timeout
        docker.errors.DockerException
            If Docker operations fail
        """

        # Check if Docker is available
        if not self._is_docker_running():
            raise RuntimeError(
                "Docker is not running. Please start Docker or install Ollama manually.")

        # -> Container already running
        # Check if container exists and is running and restart if requested
        if self._is_container_running():
            if exists_ok:
                if restart:
                    print("Restarting existing Ollama container...")
                    self.stop()
                    self._start_container()
                    self._wait_for_ready()
                    return True
            else:
                if force:
                    print("Stopping existing Ollama container...")
                    self.stop()
                    print("Deleting existing Ollama container...")
                    self.delete(force=True, remove_volumes=True)
                else:
                    raise RuntimeError(
                        f"Container {self.container_name} is already running. Use exists_ok=True to allow this."
                    )

        # -> Container not running but ollama is reachable (probably locall installed)
        if self._is_port_open() and attach_port:
            print("Ollama is already running and ready")
            return False

        # -> Port is in use by a different container or process
        if self.is_port_in_use(self.port, self.host) and force:
            print(f"Port {self.port} is in use, stopping existing container...")
            self.clear_port(self.port)
            if self.is_port_in_use(self.port, self.host):
                raise RuntimeError(
                    f"Port {self.port} is still in use after clearing. Please check manually.")
        elif self.is_port_in_use(self.port, self.host):
            raise RuntimeError(
                f"Port {self.port} is already in use by another process or container. "
                "Please stop the conflicting service or use a different port.")

        # -> Check if container exists but is stopped
        container = self._get_container()
        if container:
            if exists_ok:
                print("Starting existing Ollama container...")
                self._start_container()
                self._wait_for_ready()
                return True
            elif not exists_ok and force:
                print("Deleting existing Ollama container...")
                self.delete(force=True, remove_volumes=True)
            else:
                raise RuntimeError(f"Container {self.container_name} exists but is not running. "
                                   "Use exists_ok=True to allow this or force=True to delete it.")

        # -> Create and start new container from scratch
        print("Creating new Ollama container...")
        self._pull_image()
        self._create_container()
        self._start_container()
        self._wait_for_ready(self.startup_timeout)
        return True

    def stop(self):
        """
        Stop the Ollama container.

        Gracefully stops the running container if it exists. Does nothing
        if no container is found.
        """
        container = self._get_container()
        if container:
            container.stop()
            print(f"Stopped Ollama container: {self.container_name}")

    @property
    def api_url(self) -> str:
        """
        Get the base API URL for Ollama.

        Returns
        -------
        str
            The base URL for Ollama API access
        """
        return f"http://{self.host}:{self.port}"

    @property
    def openai_compatible_url(self) -> str:
        """
        Get the OpenAI-compatible API URL for Ollama.

        Returns
        -------
        str
            The OpenAI-compatible API endpoint URL
        """
        return f"http://{self.host}:{self.port}/v1"
