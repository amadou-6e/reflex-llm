import os
import docker
import requests
import time
from typing import Optional
from pathlib import Path
# -- Ours --
from reflex_llms.settings import *


class ContainerHandler:
    """
    Handles LocalAI Docker container operations.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        image: str = "ollama/ollama:latest",
        container_name: str = "ollama-openai-backend",
        data_path: Optional[Path] = None,
        startup_timeout: int = 120,
    ):
        """
        Initialize Ollama container handler.
        
        Args:
            host: Ollama host
            port: Ollama port
            image: Docker image to use
            container_name: Name for the container
            data_path: Path to store Ollama data
            startup_timeout: Timeout for container startup
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
        """Check if Docker daemon is running."""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def _is_port_open(self) -> bool:
        """Check if Ollama port is responding."""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _get_container(self):
        """Get container by name."""
        if not self.client:
            return None
        try:
            return self.client.containers.get(self.container_name)
        except docker.errors.NotFound:
            return None

    def _is_container_running(self) -> bool:
        """Check if container exists and is running."""
        container = self._get_container()
        if not container:
            return False
        container.reload()
        return container.status == "running"

    def _pull_image(self):
        """Pull Ollama image if not present."""
        if not self.client:
            raise RuntimeError("Docker client not available")

        try:
            self.client.images.get(self.image)
        except docker.errors.NotFound:
            print(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)

    def _create_container(self):
        """Create Ollama container."""
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
        """Start the Ollama container."""
        container = self._get_container()
        if not container:
            raise RuntimeError("Container not found")

        container.start()
        print(f"Started Ollama container: {self.container_name}")

    def _wait_for_ready(self, timeout: int = 120):
        """Wait for Ollama to be ready."""
        print("Waiting for Ollama to be ready...")
        start_time = time.time()

        while (timeout is None) or (time.time() - start_time < timeout):
            if self._is_port_open():
                print("Ollama is ready")
                return True
            time.sleep(3)

        raise ConnectionError(f"Ollama did not become ready within {timeout} seconds")

    def ensure_running(self):
        """Ensure Ollama container is running and ready."""
        # Check if port is already open
        if self._is_port_open():
            print("Ollama is already running and ready")
            return

        # Check if Docker is available
        if not self._is_docker_running():
            raise RuntimeError(
                "Docker is not running. Please start Docker or install Ollama manually.")

        # Check if container exists and is running
        if self._is_container_running():
            print("Container is running but not ready, waiting...")
            self._wait_for_ready()
            return

        # Check if container exists but is stopped
        container = self._get_container()
        if container:
            print("Starting existing Ollama container...")
            self._start_container()
            self._wait_for_ready()
            return

        # Create and start new container
        print("Creating new Ollama container...")
        self._pull_image()
        self._create_container()
        self._start_container()
        self._wait_for_ready(self.startup_timeout)

    def stop(self):
        """Stop the Ollama container."""
        container = self._get_container()
        if container:
            container.stop()
            print(f"Stopped Ollama container: {self.container_name}")

    @property
    def api_url(self) -> str:
        """Get the API URL for Ollama."""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_compatible_url(self) -> str:
        """Get the OpenAI-compatible API URL for Ollama."""
        return f"http://{self.host}:{self.port}/v1"
