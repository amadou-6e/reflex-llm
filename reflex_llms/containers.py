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
        port: int = 8080,
        image: str = "localai/localai:latest-cpu",
        container_name: str = "methodsheet-localai",
        models_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        startup_timeout: int = 120,
        rebuild: bool = False,
        preload: bool = False,
        config_file: Optional[Path] = None,
        n_threads: int = 4,
    ):
        """
        Initialize LocalAI container handler.
        
        Args:
            host: LocalAI host
            port: LocalAI port
            image: Docker image to use
            container_name: Name for the container
            models_path: Path to store models
        """
        self.host = host
        self.port = port
        self.image = image
        self.rebuild = rebuild
        self.preload = preload
        self.container_name = container_name
        self.n_threads = n_threads
        self.models_path = models_path or MODEL_PATH
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path or DEFAULT_CACHE
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.config_path = config_file or CONFIG_PATH
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.config_file = config_file or Path(CONFIG_PATH, "localai.yml")
        self._copy_default_config()

        self.startup_timeout = startup_timeout

        self.client = None
        if self._is_docker_running():
            self.client = docker.from_env()

    def _copy_default_config(self):
        """Copy default config file if not present."""
        if not self.config_file.exists():
            default_config = Path(DEFAULT_CONFIG_PATH, "localai.yml")
            if default_config.exists():
                with open(default_config, "r") as src:
                    with open(self.config_file, "w") as dst:
                        dst.write(src.read())
            else:
                raise FileNotFoundError(f"Default config file not found: {default_config}")

    def _is_docker_running(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def _is_port_open(self) -> bool:
        """Check if LocalAI port is responding."""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/readyz", timeout=5)
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
        """Pull LocalAI image if not present."""
        if not self.client:
            raise RuntimeError("Docker client not available")

        try:
            self.client.images.get(self.image)
        except docker.errors.NotFound:
            print(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)

    def _create_container(self):
        """Create LocalAI container."""
        if not self.client:
            raise RuntimeError("Docker client not available")

        # Remove existing container if it exists
        existing = self._get_container()
        if existing:
            existing.remove(force=True)

        mounts = [
            docker.types.Mount(
                target="/build/models",
                source=str(self.models_path.resolve()),
                type="bind",
            ),
            docker.types.Mount(
                target="/tmp/localai",
                source=str(self.cache_path.resolve()),
                type="bind",
            ),
            # docker.types.Mount(
            #     target="/etc/localai/localai.yaml",
            #     source=str(Path(self.config_path, "localai.yaml").resolve()),
            #     type="bind",
            # )
        ]

        ports = {f"8080/tcp": self.port}

        env_vars = {
            "MODELS_PATH": "/build/models",
            "LOCALAI_THREADS": self.n_threads,
            "DEBUG": True,
        }
        env_vars["REBUILD"] = "true" if self.rebuild else "false"
        env_vars["PRELOAD"] = "true" if self.preload else "false"

        container = self.client.containers.create(
            image=self.image,
            name=self.container_name,
            mounts=mounts,
            ports=ports,
            detach=True,
            environment=env_vars,
        )
        return container

    def _start_container(self):
        """Start the LocalAI container."""
        container = self._get_container()
        if not container:
            raise RuntimeError("Container not found")

        container.start()
        print(f"Started LocalAI container: {self.container_name}")

    def _wait_for_ready(self, timeout: int = 120):
        """Wait for LocalAI to be ready."""
        print("Waiting for LocalAI to be ready...")
        start_time = time.time()

        while (timeout is None) or (time.time() - start_time < timeout):
            if self._is_port_open():
                print("LocalAI is ready")
                return True
            time.sleep(3)

        raise ConnectionError(f"LocalAI did not become ready within {timeout} seconds")

    def ensure_running(self):
        """Ensure LocalAI container is running and ready."""
        # Check if port is already open
        if self._is_port_open():
            print("LocalAI is already running and ready")
            return

        # Check if Docker is available
        if not self._is_docker_running():
            raise RuntimeError(
                "Docker is not running. Please start Docker or install LocalAI manually.")

        # Check if container exists and is running
        if self._is_container_running():
            print("Container is running but not ready, waiting...")
            self._wait_for_ready()
            return

        # Check if container exists but is stopped
        container = self._get_container()
        if container:
            print("Starting existing LocalAI container...")
            self._start_container()
            self._wait_for_ready()
            return

        # Create and start new container
        print("Creating new LocalAI container...")
        self._pull_image()
        self._create_container()
        self._start_container()
        self._wait_for_ready(self.startup_timeout)

    def stop(self):
        """Stop the LocalAI container."""
        container = self._get_container()
        if container:
            container.stop()
            print(f"Stopped LocalAI container: {self.container_name}")

    @property
    def api_url(self) -> str:
        """Get the API URL for LocalAI."""
        return f"http://{self.host}:{self.port}/v1"
